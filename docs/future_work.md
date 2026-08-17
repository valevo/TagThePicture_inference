# Future Work

_current version: 11.08.2026_

## Preliminaries

 - Two basic types of mistakes are typically distinguished: false positives (related to _precision_), which is when tag is applied to an image that is wrong, and false negatives (related to _recall_), which means that a tag wasn't applied that would have been correct. The two types are in a proper trade-off, since applying more tags automatically leads to more false positives and fewer false negatives, and vice-versa. The former type is the "worse" kind of mistake because it leads to misleading information in the collection, but also somewhat "easier" to detect and estimate. The latter, despite being relatively "unproblematic" because tagging is somewhat "optional", is very hard to estimate.
   
 - There is quite a lot of inherent ambiguity in the tagging task -- for example, a "Ruin" could also be a "Palace", "Temple", etc. On the opposite, "River" and "Flood" are conceptually mutually exclusive but may visually be impossible to distinguish. This is reflected in the relatively high annotator-disagreement in the tags from the [TagThePicture platform](https://tagthepicture.nl/); when humans are not sure or don't agree on whether a certain tag applies or does not apply, neither can an automated system.
   
 - The tag distribution is very uneven, which means that some tags apply to significant amounts of the collections while other occur just a handful of times. And that's entirely natural because some tags are broader than other and some more naturally occurring concepts than others. (E.g. "tennis court" vs "Natural landscape") This implies, among other things, that accuracy scores such as false-positive and false-negative scores are skewed and we can really only reliably estimate how well the model is doing on the more frequent tags. 

## Thesaurus

Tags are (with a few exceptions) directly taken from the Wereldmuseum's [thesaurus](https://collectie.wereldmuseum.nl/thesaurus). Every entry in the thesaurus is a concept, which has on or more labels mainly in Dutch and English, and has one parent (broader) concept and potentially one or more child (narrower) concepts. In the initial step, English-language labels have been literally given to the language models, and the thesaurus' structure been discarded. It is worth noting, that (1) concepts seldomly have more than 1 (meaningfully different) label and (2) the tag set of TagThePicture contains only few internal relationships from the thesaurus.

Ways to improve the tag set itself can include:

 - Providing definitions of each tag to language model -- the individual labels themselves are often not understandable, too vague or not the kind of terminology that general-purpose models have been trained on. For example the tag "Procession" may be in principle well-known to a language model, but without a definition the model has no way of knowing whether the term should be interpreted narrowly (a religious or ceremonial procession) or broadly (any group of people walking together, e.g. a market crowd). A short, curated definition per tag -- drawn from the thesaurus' own scope notes where available, and written by hand otherwise -- would disambiguate this and reduce the number of cases where a correct tag is missed simply because the model guessed the wrong sense of a word.
   
 - Using tags' external structures: Regardless of the language model's scores, we could _automatically_ exclude specific co-occurrences of tags. That is, because of mutual semantic exclusivity of certain tags (e.g. "Mosque" vs "Church"). Conversely, we could _tentatively_ (especially if scores are high but below threshold) infer the presence of other tags given a certain tag using non-hierarchical associations in the thesaurus. Think of "Wedding" and "Group portrait" or "Coast" and "Body of Water".

 - Using tags' internal structures: If a tag applies which semantically subsumes another, then a high-confidence score on the narrower (child) concept can be used as evidence for the broader (parent) concept, even when the parent's own score falls just short of the threshold. A confident "Mosque" prediction can make "Religious building" more likely to apply as well. Propagating scores upward through the thesaurus hierarchy in this way is a targeted way of reducing false negatives on broader tags without touching the threshold used for the narrower tags.

 - External thesauri could provide alternative -- possibly culturally more precise or historically more accurate -- labels for the concepts in the tag set. 

## Language Models

The language models used for the scene-tagging and object-detection tasks so far have been [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384) and [OWLv2](https://huggingface.co/google/owlv2-base-patch16) respectively. Despite being relatively small variants of their families (due to computational limitations), these are rather standard choices, and we have been using them as-is.

 - Fine-tuning on domain data: both models were pretrained predominantly on contemporary, colour, web-sourced photographs. This likely leads to the model "overlooking" the presence of tags whose visual appearance the model has never seen in this domain. Fine-tuning (or at minimum lightweight adapter-tuning) on a sample of the TagThePicture collection, potentially using the platform's own crowd-sourced annotations as (weak) supervision, should directly target the recall gap.
   
 - Prompting currently is done by simply replacing "X" in a sentence like "a photograph of X" with each tag and using the resulting score. Prompts could be enhanced by sourcing multiple natural language sentence per tag (possibly including natural nuances related to that tag) and having the model score each of them. Such phrases could be hand-made or automatically scraped from Wikipedia and similar thesauri and encyclopedias. For example, "a photo of a procession", "a religious procession", "people marching in a procession", etc. The resulting variance in scores of multiple prompts per tag can make scoring more robust and allow for more intricate decision functions.

 - The scene-detection and object-detection models could in principle assist each other, because the scene and object tags are quite often correlated, and could lower both false-negatives as well as false-positives. The object tag "Train rails/tracks" can inform the presence of the scene tag "Railroad" and vice-versa when one model missed it. Conversely, when the scene tag "Volcano" has been detected, the object tag "Bicycle" becomes unlikely to apply. How to implement this in practice without exploding computational demands would be a matter of research but a cheap version could be part of the decision functions. 


## Decision Functions

 - tailored to the language models' outputs: 


