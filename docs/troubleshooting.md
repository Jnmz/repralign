# Troubleshooting

## The model loads but layer names fail

Use `repralign list-layers` first and copy exact names from the output.

## The SD3 reference path is too heavy for my machine

Try reducing:

- image resolution
- batch size
- number of hooked layers
- precision, if your hardware supports it

You can also start with a smaller semantic-only experiment to validate the pipeline before adding the generation reference.

## The generated plots do not look like the paper figures I expect

That usually means one of the following is different:

- the candidate model
- the reference model checkpoint
- the dataset
- the prompt setup
- the selected layers
- the metric or `k` setting

## I only have image files, not prompts

Use an image-folder dataset with an empty prompt for semantic references. For generation-oriented references, decide whether:

- empty prompts are acceptable for your experiment
- you want a constant default prompt
- you want a manifest file with per-image prompts

## The docs site builds locally but not on GitHub

Check the `docs.yml` workflow logs and make sure GitHub Pages is enabled for the repository.
