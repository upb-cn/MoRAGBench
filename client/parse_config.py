import json
import os
import sys
from typing import Any, Dict, Union

from pydantic import BaseModel, ValidationError

from classes.benchmark import TaskBenchmark, ANNBenchmark
from classes.common import SamplingMethod, TestType

# ----------------------------
# Helpers
# ----------------------------
def apply_override(cfg: dict, key: str, value: str):
    """Apply dotted override into nested dict, trying JSON parsing for value."""
    try:
        parsed = json.loads(value)
    except Exception:
        parsed = value
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        if p not in d or not isinstance(d[p], dict):
            d[p] = {}
        d = d[p]
    d[parts[-1]] = parsed


def is_path_like(s: str) -> bool:
    """Heuristic if a string is likely a filesystem path."""
    if not isinstance(s, str):
        return False
    # heuristics: contains os separator or starts with '.' or ~ or is absolute
    if os.path.sep in s or s.startswith(".") or s.startswith("~") or os.path.isabs(s):
        return True
    # also treat strings that actually exist as path-like
    return os.path.exists(os.path.expanduser(s))


def check_path(path: str) -> Union[None, str]:
    """Return None if OK else an error message."""
    p = os.path.expanduser(path)
    if not os.path.exists(p):
        return f"path does not exist: {path}"
    if os.path.isdir(p):
        try:
            # non-empty directory check
            if not any(os.scandir(p)):
                return f"directory is empty: {path}"
        except PermissionError:
            return f"permission denied reading directory: {path}"
    # file is fine
    return None

# ----------------------------
# Default-reporting logic
# ----------------------------
def report_defaults(model: BaseModel, original_cfg: Dict[str, Any], prefix: str = ""):
    """
    Recursively print any field that the user did NOT specify (i.e. missing in original_cfg)
    and the default value used in the model instance — but *only* if that field will actually
    be used at runtime.

    Special-case logic:
      - 'overlap_size' is printed only when the same model instance has attribute overlap_enabled == True
      - 'temp', 'top_k', 'top_p', 'repetition_penalty' are printed only when the same model instance
        has attribute use_sampling == True
      - 'seed' is printed only if 'sampling_method' == 'random'
    """
    # Define which conditional fields should be suppressed when their enabling flag is False
    llm_sampling_fields = {"temp", "top_k", "top_p", "repetition_penalty"}
    conditional_field_checks = {
        "overlap_size": ("overlap_enabled", True),
        "seed": ("sampling_method", SamplingMethod.RANDOM),
        # LLM sampling fields share the same controller 'use_sampling'; we'll check membership below.
    }

    # Use class-level model_fields to avoid Pydantic deprecation warning
    for name, _ in model.__class__.model_fields.items():
        value = getattr(model, name)

        # Determine if the user provided this field in the original config at this nested level
        present = isinstance(original_cfg, dict) and (name in original_cfg)

        field_path = f"{prefix}.{name}" if prefix else name

        # If the field is conditional, decide whether it's actually used or not
        should_print_conditional_default = True
        if name in conditional_field_checks:
            controller_attr, required_val = conditional_field_checks[name]
            # If the model has the controller attribute, only print default when it equals required_val
            if hasattr(model, controller_attr):
                try:
                    controller_val = getattr(model, controller_attr)
                    if controller_val != required_val:
                        should_print_conditional_default = False
                except Exception:
                    # be conservative: if we can't read controller, assume it's used
                    should_print_conditional_default = True

        # LLM sampling fields share a controller attribute named 'use_sampling'
        if name in llm_sampling_fields:
            if hasattr(model, "use_sampling"):
                try:
                    if not getattr(model, "use_sampling"):
                        should_print_conditional_default = False
                except Exception:
                    should_print_conditional_default = True

        if not present:
            # Only print the missing-field default if it's not a conditional field that is unused
            if should_print_conditional_default:
                print(f"[default] {field_path} = {repr(value)}")

            # If it's a nested model, recursively report defaults for its children (but pass original_cfg=None)
            if isinstance(value, BaseModel):
                # If we didn't print the parent default due to condition, still recurse only if the parent
                # itself will be used; e.g., if overlap_enabled == False we probably don't need to descend into
                # overlap_size (it's not a nested model), but recursion is safe for other fields.
                report_defaults(value, original_cfg=None, prefix=field_path)

            elif isinstance(value, list):
                # For lists of nested models, recurse into elements
                for idx, elem in enumerate(value):
                    if isinstance(elem, BaseModel):
                        report_defaults(elem, original_cfg=None, prefix=f"{field_path}[{idx}]")
        else:
            # User provided this field. If it's a nested model, recurse and pass the nested original sub-dict.
            if isinstance(value, BaseModel):
                sub_cfg = original_cfg.get(name) if isinstance(original_cfg, dict) else None
                report_defaults(value, original_cfg=sub_cfg, prefix=field_path)
            elif isinstance(value, list):
                sub_cfg_list = original_cfg.get(name) if isinstance(original_cfg, dict) else None
                if isinstance(sub_cfg_list, list):
                    for idx, elem in enumerate(value):
                        if isinstance(elem, BaseModel):
                            sub_cfg = sub_cfg_list[idx] if idx < len(sub_cfg_list) and isinstance(sub_cfg_list[idx], dict) else None
                            report_defaults(elem, original_cfg=sub_cfg, prefix=f"{field_path}[{idx}]")
                else:
                    # user provided something non-list or didn't provide nested dicts: still recurse into defaults
                    for idx, elem in enumerate(value):
                        if isinstance(elem, BaseModel):
                            report_defaults(elem, original_cfg=None, prefix=f"{field_path}[{idx}]")


# ----------------------------
# Ignored-reporting logic
# ----------------------------
def report_ignored_user_fields(model: TaskBenchmark, raw_cfg: dict):
    """
    Report fields the user explicitly provided, but which will not be used due to:
      - overlap_enabled == False        → overlap_size ignored
      - use_sampling == False           → sampling fields ignored
      - sampling_method != "random"     → seed ignored
    """

    # ----------- 1) CHUNKER CHECK -----------
    chunker_cfg = (
        raw_cfg.get("rag_pipeline", {})
                .get("embedding", {})
                .get("chunker", {})
    )

    overlap_enabled = model.rag_pipeline.embedding.chunker.overlap_enabled

    # If user supplied overlap_size but overlap_enabled=False → ignored
    if not overlap_enabled and "overlap_size" in chunker_cfg:
        print("[ignored] rag_pipeline.embedding.chunker.overlap_size was provided "
                "but overlap_enabled=False → ignoring this value.")


    # ----------- 2) LLM SAMPLING CHECK -----------
    llm_cfg = (
        raw_cfg.get("rag_pipeline", {})
                .get("llm", {})
    )

    use_sampling = model.rag_pipeline.llm.use_sampling

    sampling_fields = ["temp", "top_k", "top_p", "repetition_penalty"]

    if not use_sampling:
        for field in sampling_fields:
            if field in llm_cfg:
                print(f"[ignored] rag_pipeline.llm.{field} was provided but use_sampling=False "
                      f"→ ignoring this value.")
                
    # ----------- 3) SEED CHECK -----------
    raw_task = raw_cfg.get("downstream_task", [])
    task = model.downstream_task
    sampling_method = task.sampling_method
    
    # If user supplied overlap_size but overlap_enabled=False → ignored
    if SamplingMethod(sampling_method) != SamplingMethod.RANDOM and "seed" in raw_task:
        print(f"[ignored] downstream_task.seed was provided "
            f"but sampling_method!={SamplingMethod.RANDOM.value} → ignoring this value.")

# ----------------------------
# Main CLI logic
# ----------------------------
def validate_config(args_set, raw_cfg, test_type: TestType) -> TaskBenchmark | ANNBenchmark:

    for s in args_set:
        if "=" not in s:
            print("Overrides must be KEY=VALUE", file=sys.stderr)
            sys.exit(2)
        k, v = s.split("=", 1)
        apply_override(raw_cfg, k, v)

    if test_type == TestType.TASK:
        # Parse into pydantic model (this will run type checks)
        try:
            bench_model = TaskBenchmark(**raw_cfg)
        except ValidationError as e:
            print("Configuration validation error (Pydantic):", file=sys.stderr)
            print(e, file=sys.stderr)
            sys.exit(2)

        # Report defaults (fields the user did NOT specify)
        report_defaults(bench_model, original_cfg=raw_cfg)
        print()
            
        # Check if some params will be ignored
        report_ignored_user_fields(model=bench_model, raw_cfg=raw_cfg)
        
    else:
        try:
            bench_model = ANNBenchmark(**raw_cfg)
        except ValidationError as e:
            print("Configuration validation error (Pydantic):", file=sys.stderr)
            print(e, file=sys.stderr)
            sys.exit(2)
        
        # Report defaults (fields the user did NOT specify)
        report_defaults(bench_model, original_cfg=raw_cfg)
        print()

    print("\nAll checks passed.")
    
    return bench_model


def parse_config(args_set, raw_cfg, test_type, output_path):
    bench_model = validate_config(args_set, raw_cfg, test_type)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
    # Save config file
    with open(output_path, "w") as f:
        json.dump(bench_model.model_dump(mode="json"), f, indent=2)
        
    return bench_model