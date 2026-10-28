# Agent instructions

Analysis repo for the "Jailbreaks Live in the Persona Subspace" paper (Overleaf repo lives separately; see `handover.md` for infrastructure).

## Writing style — no AI slop

Distilled from [Wikipedia:Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing). Applies to everything written in this repo: reports in `results/`, docs, commit messages.

### Words and phrases
- No significance inflation. Banned: "stands as a testament", "plays a crucial/pivotal/vital role", "underscores the importance of", "marks a pivotal moment", "rich tapestry", "delve", metaphorical "landscape", "foster", "leverage", "enhance", "align with", "boasts", "showcase", "meticulous", "intricate", "interplay", "garner", "bolster", "groundbreaking", "renowned", "valuable insights", "diverse array".
- Say "is" and "are". Don't dodge the copula with "serves as", "functions as", "marks", "features".
- No promotional adjectives: vibrant, seamless, robust, powerful, comprehensive, cutting-edge, notable. "Significant" only with a statistic attached.
- No editorializing lead-ins: "it is important/worth noting", "interestingly", "notably", "crucially". If it matters, the content shows it.
- No vague attribution: "researchers argue", "experts suggest", "it is widely believed". Name it and cite it, or cut it.
- No false ranges ("from X to Y") unless it is literally a range.

### Sentences
- No negative parallelisms: "not just X but Y", "it's not about X, it's about Y", "not a mirror but a portal".
- No default rule-of-three. A list has exactly as many items as there are real items.
- No participle-clause tails that smuggle in unsupported analysis: "..., highlighting ...", "..., ensuring ...", "..., reflecting ...". A real claim gets a main clause and evidence.
- Repeat technical terms verbatim. No elegant variation: a trait vector is "trait vector" every time, not "persona direction/axis/dimension" in rotation.
- Em dashes sparingly, not as universal punctuation glue.
- At most one hedge per claim: "may", not "may potentially suggest".

### Structure
- No boilerplate wrap-ups: "In conclusion", "In summary", "Overall", "Despite these challenges, ...". End when the content ends.
- Every sentence must carry information the reader does not already have. If a sentence could sit unchanged in any other report, delete it or sharpen it.
- Don't replace argument with bullet lists; bullet walls of `**Bold header**: text` are a slop signature. Tables are for tabular data, prose carries the reasoning.

### Formatting
- Headings in sentence case. No emoji. Bold a term at its first definition only.
- No thematic-break rulers between every section.

### Facts, citations, numbers
- Every number traces to a result file or script output; name the source next to the number.
- Never invent references. Check every DOI/arXiv ID resolves to the claimed paper. Strip `utm_`/tracking parameters from URLs.
- No knowledge-cutoff hedging ("as of 2026, ...") unless data vintage is the point.

### Tone
- No sycophancy, no letter-writing ("I hope this helps"), no self-congratulation in commit messages ("ensured adherence to ..."). State what changed, plainly.
- Every "selling" sentence must be anchored to a number, table, or citation. Superlatives and surprise markers ("remarkably", "strikingly", "surprisingly") at most once per report, and only when the result contradicts prior expectations stated in writing.

## Results structure
- Report prose is claim-led: a short bold claim, then one supporting sentence, then the numbers (in a table if more than a couple). Name the producing script next to the numbers.
