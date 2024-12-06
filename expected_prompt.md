## Example of {paper_data} ( Retreive data from vectorDB )
===================================================================================================================

```json
{
    "id": "2410.18541_arxiv",
    "title": "On Explaining with Attention Matrices",
    "type": "paper",
    "year": 2024,
    "abstract": "  This paper explores the much discussed, possible explanatory link between\nattention weights (AW) in transformer models and predicted output. Contrary to\nintuition and early research on attention, more recent prior research has\nprovided formal arguments and empirical evidence that AW are not explanatorily\nrelevant. We show that the formal arguments are incorrect. We introduce and\neffectively compute efficient attention, which isolates the effective\ncomponents of attention matrices in tasks and models in which AW play an\nexplanatory role. We show that efficient attention has a causal role (provides\nminimally necessary and sufficient conditions) for predicting model output in\nNLP tasks requiring contextual information, and we show, contrary to [7], that\nefficient attention matrices are probability distributions and are effectively\ncalculable. Thus, they should play an important part in the explanation of\nattention based model behavior. We offer empirical experiments in support of\nour method illustrating various properties of efficient attention with various\nmetrics on four datasets.\n",
    "authors": [
        "Omar Naim and Nicholas Asher"
    ],
    "source": "arxiv"
}
```

==========================================================================================================
```json
{
    "id": "2204.13154_arxiv",
    "title": "Attention Mechanism in Neural Networks: Where it Comes and Where it Goes",
    "type": "paper",
    "year": 2022,
    "abstract": "  A long time ago in the machine learning literature, the idea of incorporating\na mechanism inspired by the human visual system into neural networks was\nintroduced. This idea is named the attention mechanism, and it has gone through\na long development period. Today, many works have been devoted to this idea in\na variety of tasks. Remarkable performance has recently been demonstrated. The\ngoal of this paper is to provide an overview from the early work on searching\nfor ways to implement attention idea with neural networks until the recent\ntrends. This review emphasizes the important milestones during this progress\nregarding different tasks. By this way, this study aims to provide a road map\nfor researchers to explore the current development and get inspired for novel\napproaches beyond the attention.\n",
    "authors": [
        "Derya Soydaner"
    ],
    "source": "arxiv"
}
```


## Example expected output

```json
{
    "GraphNodes" : [
        {
            data : {
                "id": "2410.18541_arxiv",
                "title": "On Explaining with Attention Matrices",
                "type": "paper",
                "year": 2024,
                "abstract": "  This paper explores the much discussed, possible explanatory link between\nattention weights (AW) in transformer models and predicted output. Contrary to\nintuition and early research on attention, more recent prior research has\nprovided formal arguments and empirical evidence that AW are not explanatorily\nrelevant. We show that the formal arguments are incorrect. We introduce and\neffectively compute efficient attention, which isolates the effective\ncomponents of attention matrices in tasks and models in which AW play an\nexplanatory role. We show that efficient attention has a causal role (provides\nminimally necessary and sufficient conditions) for predicting model output in\nNLP tasks requiring contextual information, and we show, contrary to [7], that\nefficient attention matrices are probability distributions and are effectively\ncalculable. Thus, they should play an important part in the explanation of\nattention based model behavior. We offer empirical experiments in support of\nour method illustrating various properties of efficient attention with various\nmetrics on four datasets.\n",
                "authors": [
                    "Omar Naim and Nicholas Asher"
                ],
                "source": "arxiv"
            }
        },
        {
            data : {
                "id": "2204.13154_arxiv",
                "title": "Attention Mechanism in Neural Networks: Where it Comes and Where it Goes",
                "type": "paper",
                "year": 2022,
                "abstract": "  A long time ago in the machine learning literature, the idea of incorporating\na mechanism inspired by the human visual system into neural networks was\nintroduced. This idea is named the attention mechanism, and it has gone through\na long development period. Today, many works have been devoted to this idea in\na variety of tasks. Remarkable performance has recently been demonstrated. The\ngoal of this paper is to provide an overview from the early work on searching\nfor ways to implement attention idea with neural networks until the recent\ntrends. This review emphasizes the important milestones during this progress\nregarding different tasks. By this way, this study aims to provide a road map\nfor researchers to explore the current development and get inspired for novel\napproaches beyond the attention.\n",
                "authors": [
                    "Derya Soydaner"
                ],
                "source": "arxiv"
            }
        },
        {
            // Keyword Node added
            data: {
                "id" : "transformer-and-attention-is-all-you-need",
                "title" : "Transformer and Attention is all you need",
                "type" : "keyword",
                "abstract" : "",
                "authors" : "",
                "source" : ""
            }
        }
    ],
    "GraphLinks" : [
        {
            "source" : "2410.18541_arxiv",
            "target" : "2204.13154_arxiv"
        },
        {
            "source" : "transformer-and-attention-is-all-you-need",
            "target" : "2204.13154_arxiv"
        },
        {
            "source" : "transformer-and-attention-is-all-you-need",
            "target" : "2410.18541_arxiv"
        },
    ]
}
```