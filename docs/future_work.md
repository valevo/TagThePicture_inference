# Future Work

_current version: 11.08.2026_

## Preliminaries

 - Two basic types of mistakes are typically distinguished: false positives (related to _precision_), which is when tag is applied to an image that is wrong, and false negatives (related to _recall_), which means that a tag wasn't applied that would have been correct. The two types are in a proper trade-off, since applying more tags automatically leads to more false positives and fewer false negatives, and vice-versa. The former type is the "worse" kind of mistake because it leads to misleading information in the collection, but also somewhat "easier" to detect and estimate. The latter, despite being relatively "unproblematic" because tagging is somewhat "optional", is very hard to estimate.
 - There is quite a lot of inherent ambiguity in the tagging task -- for example, a "Ruin" could also be a "Palace", "Temple", etc. On the opposite, "River" and "Flood" are conceptually mutually exclusive but may visually be impossible to distinguish. This is reflected in the relatively high annotator-disagreement in the tags from the [TagThePicture platform](https://tagthepicture.nl/); when humans are not sure or don't agree on whether a certain tag applies or does not apply, neither can an automated system.
 - 

## Thesaurus

Tags are (with a few exceptions) directly taken from the Wereldmuseum's [thesaurus](https://collectie.wereldmuseum.nl/thesaurus). Every entry in the thesaurus is a concept, which has on or more labels mainly in Dutch and English, and has one parent (broader) concept and potentially one or more child (narrower) concepts. In the initial step, English-language labels have been literally given to the language models, and the thesaurus' structure been discarded. It is worth noting, that (1) concepts seldomly have more than 1 (meaningfully different) label and (2) the tag set of TagThePicture contains only few internal relationships from the thesaurus.

Ways to improve the tag set itself can include:

 - Providing definitions of each tag to language model -- the individual labels themselves are often not understandable, too vague or not the kind of terminology that general-purpose models have been trained on. For example the tag "Procession" may be in principle well-
  -> this would improve the language-model's input and therefore the scores from it

 - Using tags' external structures: Regardless of the language model's scores, we could _automatically_ exclude specific co-occurrences of tags. That is, because of the

 - Using tags' internal structures: If a tag applies to semantically subsumes another, then 

## Language Models

The language models used for the scene-tagging and object-detection tasks so far have been [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384) and [OWLv2](https://huggingface.co/google/owlv2-base-patch16) respectively. Despite being relatively small variants of their families (due to computational limitations), these are rather standard choices, and we have been using them as-is.

 - fine tuning


## Decision Functions

 - tailored to the language models' outputs: 


