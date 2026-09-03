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
    DownstreamTaskName.HOTPOT_QA: ("hotpotqa/hotpot_qa", "distractor", "validation"),
    DownstreamTaskName.DROP: ("ucinlp/drop", "", "validation"),
    DownstreamTaskName.NATURAL_QUESTIONS: ("natural_questions", "dev", "validation"),
    DownstreamTaskName.MS_MARCO: ("microsoft/ms_marco", "v1.1", "validation"),
    DownstreamTaskName.SEARCH_QA: ("lucadiliello/searchqa", "", "validation")
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

    elif name == DownstreamTaskName.DROP:
        for item in tqdm(sampled_items, desc=f"Parsing questions for {name.value}"):
            questions.append(item["question"])
            references.append(item["answers_spans"]["spans"])

        doc_id = 0
        for item in tqdm(items_for_corpus, desc=f"Parsing documents for {name.value}"):
            passage = item["passage"]

            if passage not in documents_object.values():
                documents_object[f"doc_{doc_id}"] = passage
                doc_id += 1

    elif name == DownstreamTaskName.NATURAL_QUESTIONS:
        # Natural Questions contains examples without short-answer annotations.
        # Per Huzaifa's confirmation, we use the "text" field inside
        # short_answers as the reference (this is the only field in the
        # annotations that actually contains readable text; long_answer only
        # has token positions, no text). Examples with no short answer text
        # are skipped since the current evaluation expects text references.
        for item in tqdm(sampled_items, desc=f"Parsing questions for {name.value}"):
            answer_texts = []
            for answer in item["annotations"]["short_answers"]:
                answer_texts.extend(answer["text"])

            answer_texts = [answer for answer in answer_texts if answer.strip()]

            if not answer_texts:
                continue

            questions.append(item["question"]["text"])
            references.append(answer_texts)

        task.limit = len(questions)

        doc_id = 0
        for item in tqdm(items_for_corpus, desc=f"Parsing documents for {name.value}"):
            tokens = item["document"]["tokens"]["token"]
            is_html = item["document"]["tokens"]["is_html"]
            # Remove HTML markup tokens and keep only the readable document text
            # before adding it to the retrieval corpus.
            document_text = " ".join(
                token for token, html_flag in zip(tokens, is_html)
                if not html_flag
            )

            if document_text not in documents_object.values():
                documents_object[f"doc_{doc_id}"] = document_text
                doc_id += 1

    elif name == DownstreamTaskName.MS_MARCO:
        # MS MARCO already provides plain-text answers and passages, so no
        # extra parsing (like token-joining or HTML filtering) is needed here.
        # We use all available answers as references and all passages as the
        # retrieval corpus, consistent with how DROP and Natural Questions
        # are handled (rather than filtering to only the "is_selected" passage).
        for item in tqdm(sampled_items, desc=f"Parsing questions for {name.value}"):
            answer_texts = [answer for answer in item["answers"] if answer.strip()]

            if not answer_texts:
                continue

            questions.append(item["query"])
            references.append(answer_texts)

        task.limit = len(questions)

        doc_id = 0
        for item in tqdm(items_for_corpus, desc=f"Parsing documents for {name.value}"):
            for passage_text in item["passages"]["passage_text"]:
                if passage_text not in documents_object.values():
                    documents_object[f"doc_{doc_id}"] = passage_text
                    doc_id += 1

    elif name == DownstreamTaskName.SEARCH_QA:
        # SearchQA's context text contains leftover markup tags from how it
        # was scraped ([DOC], [TLE], [PAR]) mixed into otherwise plain text.
        # We strip these out before using the text, similar in spirit to the
        # HTML-token filtering used for Natural Questions. Every row in this
        # dataset has a non-empty answers list, so no skipping is needed here.
        def clean_searchqa_text(raw_text: str) -> str:
            for tag in ["[DOC]", "[TLE]", "[PAR]"]:
                raw_text = raw_text.replace(tag, " ")
            return " ".join(raw_text.split())

        for item in tqdm(sampled_items, desc=f"Parsing questions for {name.value}"):
            answer_texts = [answer for answer in item["answers"] if answer.strip()]

            if not answer_texts:
                continue

            questions.append(item["question"])
            references.append(answer_texts)

        task.limit = len(questions)

        doc_id = 0
        for item in tqdm(items_for_corpus, desc=f"Parsing documents for {name.value}"):
            document_text = clean_searchqa_text(item["context"])

            if document_text not in documents_object.values():
                documents_object[f"doc_{doc_id}"] = document_text
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
