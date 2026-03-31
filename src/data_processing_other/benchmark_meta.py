

BENCHMARK_DESCRIPTIONS = {
    "winogrande":{
        "lang": "en",
        "format_fn": "format_winogrande",
        "text_col": "sentence",
        "label_col": "label",
        "label_semantics": "options",
        "num_labels": 2,
        "suffixes":[
                "Choose the word that best completes the sentence using common sense reasoning.",
                "Select the option that makes the most logical sense in this context.",
                "Determine which word fits better based on real-world knowledge and common sense.",
                "Pick the word that creates the most coherent and sensible sentence.",
                "Choose the option that demonstrates proper understanding of the situation described.",
                "Select the word that best reflects common sense and everyday knowledge.",
                "Determine which choice makes the sentence more realistic and plausible.",
                "Pick the option that shows better understanding of cause and effect relationships.",
                "Choose the word that creates a more logical and reasonable statement.",
                "Select the option that better aligns with typical real-world scenarios.",
                "Determine which word makes the sentence more coherent and meaningful.",
                "Pick the choice that demonstrates better common sense reasoning.",
                "Choose the option that creates the most natural and sensible completion.",
                "Select the word that fits best given the context and common knowledge."
            ]

    },
    "piqa": {
        "lang": "en",
        "format_fn": "format_piqa",
        "text_col": "goal",
        "label_col": "label",
        "label_semantics": "options",
        "num_labels": 2,
        "suffixes": [
            "Select the most plausible solution to accomplish the described goal.",
            "Choose the option that best completes the task using common sense.",
            "Pick the solution that is more realistic and practical in the real world.",
            "Determine which option is the better next step to achieve the goal.",
            "Select the solution that best reflects how people typically perform this action.",
            "Choose the option that makes the most sense given the goal.",
            "Pick the solution that is more coherent and feasible.",
            "Decide which option is more likely to work to accomplish the goal.",
            "Select the option that best matches everyday physical reasoning.",
            "Choose the solution that is more sensible and appropriate for the situation."
        ],
    },
    "xstance": {
        "lang": "multi",
        "format_fn": "format_xstance",
        "context_cols": ["question"],
        "text_col": "comment",
        "label_col": "label",
        "label_semantics": "options",
        "num_labels": 2,
        "label_tags": {
            "en": ["Against", "Favor"],
            "de": ["Dagegen", "Dafür"],
            "fr": ["Contre", "Pour"],
            "it": ["Contro", "Favore"],
        },
        "label_semantics_by_lang": {
            "en": [
                "Against: The text expresses an unfavorable opinion, opposition, or disagreement with the claim.",
                "Favor: The text expresses a favorable opinion, support, or agreement with the claim.",
            ],
            "de": [
                "Dagegen: Der Text drückt eine ablehnende Meinung, Widerspruch oder Ablehnung gegenüber der Aussage aus.",
                "Dafür: Der Text drückt eine bejahende Meinung, Unterstützung oder Zustimmung zur Aussage aus.",
            ],
            "fr": [
                "Contre: Le texte exprime une opinion défavorable, une opposition ou un désaccord avec l'énoncé.",
                "Pour: Le texte exprime une opinion favorable, un soutien ou un accord avec l'énoncé.",
            ],
            "it": [
                "Contro: Il testo esprime un'opinione sfavorevole, opposizione o disaccordo con l'affermazione.",
                "Favore: Il testo esprime un'opinione favorevole, sostegno o accordo con l'affermazione.",
            ],
        },
        "suffixes": [
            "Determine whether the comment supports or opposes the question.",
            "Classify the stance expressed in the comment toward the question.",
            "Decide if the comment is in favor of or against the claim in the question.",
            "Identify whether the comment agrees or disagrees with the question's proposition.",
            "Choose the stance that best matches the comment's position.",
            "Select whether the comment expresses support or opposition.",
            "Assess the comment's stance toward the question.",
            "Pick the stance label that aligns with the comment's intent."
        ],
        "suffixes_by_lang": {
            "de": [
                "Bestimmen Sie, ob der Kommentar die Frage unterstützt oder ablehnt.",
                "Klassifizieren Sie die im Kommentar ausgedrückte Haltung zur Frage.",
                "Entscheiden Sie, ob der Kommentar die Aussage der Frage bejaht oder verneint.",
                "Bestimmen Sie, ob der Kommentar der Aussage der Frage zustimmt oder widerspricht.",
                "Wählen Sie die Haltung, die der Position des Kommentars am besten entspricht.",
                "Geben Sie an, ob der Kommentar Unterstützung oder Ablehnung ausdrückt.",
                "Bewerten Sie die Haltung des Kommentars gegenüber der Frage.",
                "Wählen Sie das Haltungslabel, das zur Absicht des Kommentars passt."
            ],
            "fr": [
                "Déterminez si le commentaire soutient ou s'oppose à la question.",
                "Classez la position exprimée dans le commentaire vis-à-vis de la question.",
                "Décidez si le commentaire est pour ou contre l'énoncé de la question.",
                "Identifiez si le commentaire est d'accord ou en désaccord avec la proposition.",
                "Choisissez la position qui correspond le mieux à l'avis du commentaire.",
                "Indiquez si le commentaire exprime un soutien ou une opposition.",
                "Évaluez la position du commentaire par rapport à la question.",
                "Sélectionnez l'étiquette de position qui correspond à l'intention du commentaire."
            ],
            "it": [
                "Determina se il commento sostiene o si oppone alla domanda.",
                "Classifica la posizione espressa nel commento rispetto alla domanda.",
                "Decidi se il commento è a favore o contro l'affermazione nella domanda.",
                "Identifica se il commento è d'accordo o in disaccordo con la proposta.",
                "Scegli la posizione che meglio corrisponde all'intento del commento.",
                "Indica se il commento esprime sostegno o opposizione.",
                "Valuta la posizione del commento rispetto alla domanda.",
                "Seleziona l'etichetta di posizione che corrisponde all'intento del commento."
            ],
        },
    },
    "semeval2016": {
        "lang": "en",
        "format_fn": "format_semeval2016",
        "context_cols": ["target"],
        "text_col": "tweet",
        "label_col": "label",
        "label_semantics": "options",
        "num_labels": 3,
        "data_files": {
            "train": "other_benchmarkts/semeval2016/trainingdata-all-annotations.txt",
            "valid": "other_benchmarkts/semeval2016/trialdata-all-annotations.txt",
            "test": "other_benchmarkts/semeval2016/testdata-taskA-all-annotations.txt",
        },
        "label_tags": ["Against", "Favor", "None"],
        "suffixes": [
            "Determine whether the tweet supports, opposes, or is neutral toward the target.",
            "Classify the stance expressed in the tweet toward the target.",
            "Decide if the tweet is in favor of, against, or neutral toward the target.",
            "Identify whether the tweet agrees, disagrees, or takes no stance on the target.",
            "Choose the stance label that best matches the tweet's position.",
            "Select whether the tweet expresses support, opposition, or neutrality.",
            "Assess the tweet's stance toward the target.",
            "Pick the stance label that aligns with the tweet's intent."
        ],
    },
    "cstance": {
        "lang": "zh",
        "format_fn": "format_cstance",
        "context_cols": ["target"],
        "text_col": "text",
        "label_col": "label",
        "label_semantics": "options",
        "num_labels": 3,
        "data_files": {
            "train": "other_benchmarks/cstance/raw_train_all_onecol.csv",
            "valid": "other_benchmarks/cstance/raw_val_all_onecol.csv",
            "test": "other_benchmarks/cstance/raw_test_all_onecol.csv",
        },
        "label_tags": {
            "en": ["Against", "Favor", "None"],
            "zh": ["反对", "支持", "中立"],
        },
        "label_semantics_by_lang": {
            "en": [
                "Against: The text expresses opposition or disagreement with the target.",
                "Favor: The text expresses support or agreement with the target.",
                "None: The text does not express a clear stance toward the target.",
            ],
            "zh": [
                "反对：文本对目标表达反对或不同意。",
                "支持：文本对目标表达支持或赞同。",
                "中立：文本未对目标表达明确立场。",
            ],
        },
        "suffixes": [
            "Determine whether the text supports, opposes, or is neutral toward the target.",
            "Classify the stance expressed in the text toward the target.",
            "Decide if the text is in favor of, against, or neutral toward the target.",
            "Identify whether the text agrees, disagrees, or takes no stance on the target.",
            "Choose the stance label that best matches the text's position.",
            "Select whether the text expresses support, opposition, or neutrality.",
            "Assess the text's stance toward the target.",
            "Pick the stance label that aligns with the text's intent."
        ],
        "suffixes_by_lang": {
            "zh": [
                "判断文本对目标是支持、反对还是中立。",
                "分类文本中对目标的立场。",
                "决定文本是支持、反对还是中立。",
                "识别文本是同意、反对还是未表态。",
                "选择最符合文本立场的标签。",
                "判断文本表达的是支持、反对还是中立。",
                "评估文本对目标的立场。",
                "选择与文本意图一致的立场标签。"
            ],
        },
    },
    "fiqa": {
        "lang": "en",
        "format_fn": "format_figqa",
        "text_col": "sentence",
        "label_col": "label",
        "label_semantics": "options",
        "num_labels": 2,
        "suffixes": [
            "Choose the ending that best matches the figurative meaning of the sentence.",
            "Select the continuation that fits the intended figurative sense.",
            "Pick the ending that best reflects the implied meaning.",
            "Determine which ending is more consistent with figurative language.",
            "Choose the option that captures the metaphorical intent.",
            "Select the ending that best matches the figurative interpretation.",
            "Pick the continuation that aligns with the implied sense.",
            "Decide which option best completes the figurative expression."
        ],
    },
    "yelp": {
        "lang": "en",
        "format_fn": "format_yelp",
        "text_col": "review",
        "label_col": "label",
        "label_semantics": "options",
        "num_labels": 2,
        "label_tags": ["Negative", "Positive"],
        "suffixes": [
            "Classify the sentiment of the review as negative or positive.",
            "Decide whether the review expresses negative or positive sentiment.",
            "Choose the sentiment label that best matches the review.",
            "Determine if the review is negative or positive overall.",
            "Select whether the review is favorable or unfavorable."
        ],
    },
    "eic": {
        "lang": "en",
        "format_fn": "format_eic",
        "context_cols": ["old"],
        "text_col": "new",
        "label_col": "label",
        "label_semantics": "options",
        "num_labels": 5,
        "label_tags": ["Claim", "Clarity", "Fact/Evidence", "Grammar", "Other"],
        "suffixes": [
            "Classify the edit intent for the source and target text.",
            "Choose the edit intent category that best describes the change.",
            "Determine which intent label matches the edit.",
            "Select the edit intent for this revision."
        ],
    }
}

# Backwards-compat: older configs used the misspelled key "sufixes".
for _benchmark, _meta in BENCHMARK_DESCRIPTIONS.items():
    if "suffixes" in _meta and "sufixes" not in _meta:
        _meta["sufixes"] = _meta["suffixes"]
