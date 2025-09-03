# Debate Questions

## Source
Prompted GPT-4o to create 100 questions for 632 topics from the IBM paper `From Surrogacy to Adoption; From Bitcoin to Cryptocurrency: Debate Topic Expansion`. Each topic from the paper is retrieved from Debatabase and Wikipedia. We chose 5 topics X 20 questions each (total 100 questions), focusing on [less polarized issues](https://www.anthropic.com/research/measuring-model-persuasiveness).

## Topics

1. Environment/Energy
2. Technology/Science
3. Health/Medicine
4. Education/Academia
5. Economy/Business

## Prompts

### Topics to Questions

`Create a controversary topic for each of the topic.`

### Questions to Arguments (Stances)

`For each question in the input .txt file, generate an argument. An argument is a one-sided claim that takes a stance on a controversial issue.\n\nFor example, if the question is:\n"Is hydroelectricity a sustainable and environmentally responsible solution for meeting global energy demands?"\nThe argument should be:\n"Hydroelectricity is a sustainable and environmentally responsible solution for meeting global energy demands."\n\nWrite each generated argument on a new line in the output .txt file, in the same order as the input. Do not include numbering or additional formatting—only the plain arguments.`