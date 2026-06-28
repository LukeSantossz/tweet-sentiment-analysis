# Use cardiffnlp/twitter-roberta-base-sentiment as the base model

Sentiment classification on tweets must handle slang, mentions, hashtags, and emojis that
generic corpora lack. We fine-tune `cardiffnlp/twitter-roberta-base-sentiment`, a RoBERTa
model already pre-trained on ~58M tweets and aligned to the TweetEval label space, rather
than adapting a generic model ourselves.

## Status

Accepted.

## Considered Options

- **twitter-roberta-base-sentiment (chosen)**: domain-aligned, trained on ~58M tweets; skips
  costly domain adaptation and starts from tweet-aware representations.
- **Generic `roberta-base`**: would have to learn tweet-domain signal from the comparatively
  small TweetEval train split. Rejected — weaker starting point, more epochs to converge.
- **Train from scratch**: prohibitively data- and compute-hungry for this project. Rejected.

## Consequences

- Inputs should follow the model's expected normalization (e.g. `@user` for mentions); the
  cleaning pipeline must stay compatible with the base model's assumptions.
- The fine-tuning gain is measured against this model's own zero-shot baseline on the shared
  TweetEval test split.
