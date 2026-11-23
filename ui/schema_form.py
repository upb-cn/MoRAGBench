"""Render Streamlit widgets straight from the client's Pydantic config classes.

Nothing about the supported options is hard-coded here: enums, defaults,
numeric bounds and nested sections are all read from ``client/classes``.
Adding a new model, dataset or index type there makes it appear in the UI
automatically.
"""

from __future__ import annotations

import json
import types
import typing
from enum import Enum
from typing import Any

import annotated_types as at
import streamlit as st
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

# Fields that are only used when another field in the same section has a
# given value. Mirrors the rules in client/parse_config.py so the form only
# shows settings that will actually take effect.
CONDITIONAL_FIELDS: dict[str, tuple[str, Any]] = {
    "seed": ("sampling_method", "random"),
    "overlap_size": ("overlap_enabled", True),
    "temp": ("use_sampling", True),
    "top_p": ("use_sampling", True),
    "repetition_penalty": ("use_sampling", True),
}
# LLM.top_k shares the name with Faiss.top_k, so it is scoped by section.
SECTION_CONDITIONAL_FIELDS: dict[tuple[str, str], tuple[str, Any]] = {
    ("llm", "top_k"): ("use_sampling", True),
}

# Starting values for required fields that have no default in client/classes
# (taken from the README / slide examples). Only used to pre-fill the widget.
SUGGESTED: dict[str, Any] = {
    "nlist": 16, "nprobe": 8, "num_training_vectors": 1000,
    "m": 32, "ef_construction": 200, "ef_search": 64,
}

# Fields that read better as a multi-line box.
LONG_TEXT_FIELDS = {"system_prompt"}
SECRET_FIELDS = {"hf_token"}

HELP: dict[str, str] = {
    "limit": "Number of questions / vectors to use. -1 = all.",
    "corpus_limit": "Number of documents in the corpus. Must be >= limit. Auto = same as limit.",
    "kv_window": "KV cache window size in tokens.",
    "max_prompt_tokens": "Prompt budget. Auto = kv_window - max_tokens.",
    "generate_until": 'Stop sequences as a JSON list, e.g. ["\\n", "\\n\\n"]. Leave empty for none.',
    "ignore_eos": "If on, generation continues past the end-of-sequence token until max_tokens.",
    "use_cache": "Reuse a serialized FAISS index on the phone if one exists.",
    "nprobe": "Clusters searched per query (<= nlist).",
    "nlist": "Number of IVF clusters (<= num_training_vectors).",
    "hf_token": "Hugging Face token, only needed for gated models. Not written to the config file if empty.",
}


# ── type helpers ────────────────────────────────────────────────────────────


def _unwrap_optional(tp):
    """Optional[X] -> (X, True); X -> (X, False)."""
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1 and len(typing.get_args(tp)) == 2:
            return args[0], True
    return tp, False


def _strip_annotated(tp):
    meta = []
    while typing.get_origin(tp) is typing.Annotated:
        args = typing.get_args(tp)
        tp, meta = args[0], meta + list(args[1:])
    return tp, meta


def _model_union_members(tp) -> list[type[BaseModel]] | None:
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(tp)
        if args and all(isinstance(a, type) and issubclass(a, BaseModel) for a in args):
            return list(args)
    return None


def _is_model(tp) -> bool:
    return isinstance(tp, type) and issubclass(tp, BaseModel)


def _is_enum(tp) -> bool:
    return isinstance(tp, type) and issubclass(tp, Enum)


def _is_str_list(tp) -> bool:
    return typing.get_origin(tp) in (list, typing.List) and typing.get_args(tp) == (str,)


def _bounds(metadata) -> tuple[float | None, float | None]:
    lo = hi = None
    for m in metadata:
        if isinstance(m, at.Gt):
            lo = m.gt + (1 if isinstance(m.gt, int) else 1e-6)
        elif isinstance(m, at.Ge):
            lo = m.ge
        elif isinstance(m, at.Lt):
            hi = m.lt
        elif isinstance(m, at.Le):
            hi = m.le
    return lo, hi


def _default_of(field) -> Any:
    if field.default is not PydanticUndefined:
        return field.default
    if field.default_factory is not None:
        return field.default_factory()
    return PydanticUndefined


def _plain(v: Any) -> Any:
    """Enum / model defaults -> JSON-style values."""
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, BaseModel):
        return v.model_dump(mode="json")
    return v


def pick_union_member(members: list[type[BaseModel]], method: str | None) -> type[BaseModel]:
    """Match faiss.method ('ivf') to its config class (IVFConfig) by name."""
    if method:
        for cls in members:
            if cls.__name__.lower().startswith(method.lower()):
                return cls
    return members[0]


# ── widget state ────────────────────────────────────────────────────────────


def _key(prefix: str, name: str) -> str:
    return f"cfg::{prefix}.{name}" if prefix else f"cfg::{name}"


def _init(key: str, value: Any):
    if key not in st.session_state:
        st.session_state[key] = value


def reset_form_state():
    for k in [k for k in st.session_state if str(k).startswith("cfg::")]:
        del st.session_state[k]


# ── rendering ───────────────────────────────────────────────────────────────


def render_model(model_cls: type[BaseModel], initial: dict | None, prefix: str = "", section: str = "") -> dict:
    """Render widgets for every field of ``model_cls``; return the config dict.

    Only fields that are required, or differ from their default, are emitted.
    That keeps generated configs short and identical in style to the
    hand-written ones in benchmark/.
    """
    initial = initial or {}
    out: dict[str, Any] = {}
    values: dict[str, Any] = {}  # current value of every field (for conditions)

    fields = list(model_cls.model_fields.items())
    # Union-typed sections (faiss.config) depend on a sibling (faiss.method),
    # so render plain fields first and nested sections last.
    simple = [(n, f) for n, f in fields if not (_is_model(f.annotation) or _model_union_members(f.annotation))]
    nested = [(n, f) for n, f in fields if (n, f) not in simple]

    cols = st.columns(3)
    col_i = 0
    for name, field in simple:
        cond = SECTION_CONDITIONAL_FIELDS.get((section, name)) or CONDITIONAL_FIELDS.get(name)
        if cond and cond[0] in values and values[cond[0]] != cond[1]:
            continue  # not used with the current settings -> hide and omit

        wide = name in LONG_TEXT_FIELDS or _is_str_list(_unwrap_optional(field.annotation)[0])
        container = st.container() if wide else cols[col_i % 3]
        if not wide:
            col_i += 1
        with container:
            value, emit = _render_field(name, field, initial, prefix)
        values[name] = value
        if emit:
            out[name] = value

    for name, field in nested:
        sub_initial = initial.get(name) if isinstance(initial.get(name), dict) else None
        sub_prefix = f"{prefix}.{name}" if prefix else name
        members = _model_union_members(field.annotation)
        if members:
            cls = pick_union_member(members, values.get("method"))
            # Different config class -> its widgets use a class-specific key
            if cls.model_fields:
                st.markdown(f"**{name}** · {cls.__name__}")
                out[name] = render_model(cls, sub_initial, f"{sub_prefix}[{cls.__name__}]", section=name)
            else:
                st.caption(f"{cls.__name__}: no parameters.")
                out[name] = {}
        else:
            st.markdown(f"**{name}**")
            # Always emit nested sections, even when empty: an omitted section
            # goes through its default_factory, which the @dataclass decorators
            # in client/classes break (e.g. Embedding.chunker -> Chunker()).
            out[name] = render_model(field.annotation, sub_initial, sub_prefix, section=name)
    return out


def _render_field(name: str, field, initial: dict, prefix: str) -> tuple[Any, bool]:
    """Return (value, should_emit)."""
    key = _key(prefix, name)
    tp, optional = _unwrap_optional(field.annotation)
    tp, ann_meta = _strip_annotated(tp)
    metadata = list(field.metadata) + ann_meta
    default = _plain(_default_of(field))
    required = field.is_required()
    start = initial.get(name, default)
    if start is PydanticUndefined:  # required field without a default (IVF/HNSW params)
        start = SUGGESTED.get(name)
    help_ = HELP.get(name) or field.description
    label = name + (" *" if required else "")

    # Optional numeric with None default -> "auto" toggle (corpus_limit, max_prompt_tokens)
    if optional and default is None and tp in (int, float):
        auto_key = key + "::auto"
        _init(auto_key, start is None)
        auto = st.toggle(f"{name}: auto", key=auto_key, help=help_)
        if auto:
            return None, False
        lo, hi = _bounds(metadata)
        num_key = key + "::num"
        _init(num_key, tp(start) if start is not None else tp(lo if lo is not None else 1))
        v = st.number_input(name, key=num_key, min_value=lo, max_value=hi, step=1 if tp is int else 0.05)
        return v, True

    if _is_enum(tp):
        options = [m.value for m in tp]
        _init(key, start if start in options else options[0])
        v = st.selectbox(label, options, key=key, help=help_)
        return v, required or v != default

    if tp is bool:
        _init(key, bool(start))
        v = st.toggle(name, key=key, help=help_)
        return v, v != default

    if tp in (int, float):
        lo, hi = _bounds(metadata)
        val = start if start is not None else (lo if lo is not None else 0)
        # Some class defaults violate their own bounds (LLM.top_k = 0 but PositiveInt);
        # clamp the widget, but don't emit the clamped value unless the user changes it.
        clamped = lo is not None and val < lo
        if clamped:
            val = lo
        _init(key, tp(val))
        v = st.number_input(
            label, key=key, min_value=tp(lo) if lo is not None else None,
            max_value=tp(hi) if hi is not None else None,
            step=1 if tp is int else 0.05, help=help_,
        )
        if clamped and v == val and name not in initial:
            return v, False
        return v, required or v != default or name in initial

    if _is_str_list(tp):
        _init(key, json.dumps(start) if start else "")
        raw = st.text_input(name, key=key, help=help_)
        if not raw.strip():
            return None, False
        try:
            parsed = json.loads(raw)
            if not (isinstance(parsed, list) and all(isinstance(s, str) for s in parsed)):
                raise ValueError
        except ValueError:
            st.error(f"{name} must be a JSON list of strings")
            return None, False
        return parsed, parsed != default

    # plain string
    _init(key, start or "")
    if name in LONG_TEXT_FIELDS:
        v = st.text_area(name, key=key, help=help_, height=90)
    else:
        v = st.text_input(name, key=key, help=help_, type="password" if name in SECRET_FIELDS else "default")
    if optional and not v:
        return None, False
    return v, required or v != default


# ── validation ──────────────────────────────────────────────────────────────


def validate(model_cls: type[BaseModel], cfg: dict) -> tuple[BaseModel | None, list[str]]:
    """Run the exact Pydantic validation the CLI runs. Return (model, errors)."""
    try:
        return model_cls(**cfg), []
    except Exception as e:  # ValidationError or ValueError from model validators
        errs = getattr(e, "errors", None)
        if callable(errs):
            msgs = []
            for err in errs():
                loc = ".".join(str(p) for p in err.get("loc", ()))
                msgs.append(f"{loc}: {err.get('msg')}" if loc else err.get("msg", str(err)))
            return None, msgs
        return None, [str(e)]
