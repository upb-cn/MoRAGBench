from classes.downstream_task import DownstreamTask, DownstreamTaskName
from datasets import load_dataset
import os
from tqdm import tqdm
from huggingface_hub.utils import HfHubHTTPError
import json
import shutil
from utils.shared import sample_items


# Tuple of (dataset_hf_path, dataset_subset, dataset_split)
DATASET_CONFIGS = {
    DownstreamTaskName.TRIVIA_QA: ("mandarjoshi/trivia_qa", "rc", "validation"),
    DownstreamTaskName.SQUAD: ("rajpurkar/squad_v2", "", "train"),
    DownstreamTaskName.HOTPOT_QA: ("hotpotqa/hotpot_qa", "distractor", "validation")
}

def parse_task(task: DownstreamTask, token: str | None, downstream_task_dir: str):
    name = task.name
    limit = task.limit
    sampling_method = task.sampling_method
    seed = task.seed
    corpus_limit = task.corpus_limit
    
    if limit < 0:
        limit = -1
        print(f"INFO: Limit for task {name} is set to a negative value. All items will be processed")
    
    # Prepare config
    chosen_config = DATASET_CONFIGS[name]
    dataset_hf_path = chosen_config[0]
    dataset_subset = chosen_config[1]
    dataset_split = chosen_config[2]
    
    # Load dataset
    try:
        items = load_dataset(
            dataset_hf_path,
            dataset_subset,
            split=dataset_split,
            use_auth_token=token
        )
    except HfHubHTTPError as e:
        # Authentication / authorization errors
        if e.response is not None and e.response.status_code in (401, 403):
            raise RuntimeError("Invalid or missing Hugging Face token") from e
        else:
            raise RuntimeError("Error downloading downstream dataset") from e
    
    # Convert to list
    items = list(items)

    # Sample from the items
    sampled_items, sampled_indices = sample_items(
        task_name=name.value,
        items=items,
        limit=limit,
        sampling_method=sampling_method,
        seed=seed
    )

    # Change limit to actual number of items
    task.limit = len(sampled_items)

    # Build items_for_corpus: always start with sampled_items, then add extra if needed
    extra_count = corpus_limit - task.limit if corpus_limit != -1 else len(items) - task.limit
    if extra_count > 0:
        sampled_set = set(sampled_indices)
        remaining_items = [item for i, item in enumerate(items) if i not in sampled_set]
        extra_items, _ = sample_items(
            task_name=name.value,
            items=remaining_items,
            limit=extra_count,
            sampling_method=sampling_method,
            seed=seed
        )
        items_for_corpus = sampled_items + extra_items
    else:
        items_for_corpus = sampled_items

    # Prepare output dir
    dir_path = f"{downstream_task_dir}/{name.value}/"
        
    # Delete if exists
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    
    # Create directory
    os.makedirs(dir_path, exist_ok=True)
    
    # Prepare variables
    documents_object = {}
    questions = []
    references = []
        
    # Prepare the corpus and save it in a document
    if name == DownstreamTaskName.TRIVIA_QA:
        for item in tqdm(sampled_items, desc=f"Parsing questions for {name.value}"):
            questions.append(item["question"])
            references.append(item['answer']['aliases'])

        # For this dataset, the corpus will be based on 
        # entity_pages.wiki_context
        doc_id = 0
        for item in tqdm(items_for_corpus, desc=f"Parsing documents for {name.value}"):
            for document in item["entity_pages"]["wiki_context"]:
                documents_object[f"doc_{doc_id}"] = document
                doc_id += 1

    elif name == DownstreamTaskName.SQUAD:
        for item in tqdm(sampled_items, desc=f"Parsing questions for {name.value}"):
            questions.append(item["question"])
            references.append(item['answers']['text'])

        doc_id = 0
        for item in tqdm(items_for_corpus, desc=f"Parsing documents for {name.value}"):
            # There are repeated contexts here. Make sure not to repeat
            context = item["context"]
            if context not in documents_object.values():
                documents_object[f"doc_{doc_id}"] = context
                doc_id += 1

    elif name == DownstreamTaskName.HOTPOT_QA:
        for item in tqdm(sampled_items, desc=f"Parsing questions for {name.value}"):
            questions.append(item["question"])
            references.append([item['answer']])

        doc_id = 0
        for item in tqdm(items_for_corpus, desc=f"Parsing documents for {name.value}"):
            for sentence in item['context']['sentences']:
                doc_text = "\n".join(sentence)
                documents_object[f"doc_{doc_id}"] = doc_text
                doc_id += 1
    else:
        raise ValueError(f"{name} is not supported yet")
    
    # Save json files
    with open(f"{dir_path}/documents.json", "w") as f:
        json.dump(documents_object, f, indent=4)
    
    with open(f"{dir_path}/questions.json", "w") as f:
        json.dump(questions, f, indent=4)
    
    with open(f"{dir_path}/references.json", "w") as f:
        json.dump(references, f, indent=4)
    
    print(
        f"\nINFO: Finished parsing downstream task: {name.value}", 
        f"\nINFO: Documents saved at: {dir_path}"
    )