
### all (100 behaviours)

| Condition | Separation harmful vs benign (folded AUC) | Paired win rate | Mean P(jb) harmful | Mean P(jb) benign |
|---|---|---|---|---|
| plain | 0.95 [0.93, 0.97] | 0.02 (n=100) | 0.101 | 0.801 |
| neutral_persona | 0.79 [0.75, 0.81] | 0.12 (n=800) | 0.290 | 0.640 |
| neutral_fiction | 0.71 [0.69, 0.73] | 0.15 (n=800) | 0.428 | 0.726 |
| jb_persona | 0.73 [0.70, 0.77] | 0.18 (n=600) | 0.310 | 0.617 |
| all_roleplay | 0.73 [0.71, 0.76] | 0.15 (n=2200) | – | – |

| Axis (all prompts of this subset) | separation (folded AUC) |
|---|---|
| harmful vs benign | 0.74 [0.72, 0.77] |
| role-play vs plain | 0.55 [0.52, 0.58] |
| role-play vs plain, harmful only | 0.82 [0.78, 0.86] |
| role-play vs plain, benign only | 0.70 [0.66, 0.74] |

| Benign role-play vs benign plain (wrapper effect) | AUC |
|---|---|
| neutral_persona | 0.70 [0.66, 0.74] |
| neutral_fiction | 0.67 [0.63, 0.71] |
| jb_persona | 0.74 [0.69, 0.78] |
| all_roleplay | 0.70 [0.66, 0.74] |

| Harmful plain vs benign role-play | AUC |
|---|---|
| neutral_persona | 0.91 [0.88, 0.95] |
| neutral_fiction | 0.92 [0.89, 0.95] |
| jb_persona | 0.90 [0.86, 0.94] |

| Frame | type | Separation harmful vs benign (folded AUC) |
|---|---|---|
| amoral_sage | jb_persona | 0.75 [0.71, 0.81] |
| dev_mode | jb_persona | 0.80 [0.75, 0.85] |
| evil_twin | jb_persona | 0.77 [0.73, 0.81] |
| opposite_day | jb_persona | 0.82 [0.78, 0.88] |
| rebel_ai | jb_persona | 0.81 [0.77, 0.85] |
| unbound | jb_persona | 0.64 [0.59, 0.68] |
| alt_history | neutral_fiction | 0.79 [0.74, 0.83] |
| fairy_tale | neutral_fiction | 0.80 [0.76, 0.86] |
| fantasy_world | neutral_fiction | 0.84 [0.79, 0.89] |
| game_quest | neutral_fiction | 0.90 [0.86, 0.94] |
| radio_drama | neutral_fiction | 0.85 [0.82, 0.89] |
| sci_fi_log | neutral_fiction | 0.53 [0.49, 0.58] |
| screenplay | neutral_fiction | 0.91 [0.87, 0.95] |
| short_story | neutral_fiction | 0.89 [0.86, 0.93] |
| grandpa | neutral_persona | 0.86 [0.82, 0.90] |
| medieval_scholar | neutral_persona | 0.87 [0.83, 0.91] |
| noir_pi | neutral_persona | 0.73 [0.69, 0.77] |
| pirate | neutral_persona | 0.85 [0.81, 0.89] |
| space_ai | neutral_persona | 0.90 [0.86, 0.94] |
| teacher | neutral_persona | 0.92 [0.89, 0.95] |
| victorian_detective | neutral_persona | 0.82 [0.77, 0.87] |
| wizard | neutral_persona | 0.86 [0.81, 0.91] |

### excl_harmbench_sourced (73 behaviours)

| Condition | Separation harmful vs benign (folded AUC) | Paired win rate | Mean P(jb) harmful | Mean P(jb) benign |
|---|---|---|---|---|
| plain | 0.94 [0.91, 0.97] | 0.02 (n=73) | 0.122 | 0.818 |
| neutral_persona | 0.79 [0.75, 0.82] | 0.11 (n=584) | 0.293 | 0.649 |
| neutral_fiction | 0.71 [0.68, 0.74] | 0.15 (n=584) | 0.427 | 0.726 |
| jb_persona | 0.73 [0.69, 0.77] | 0.18 (n=438) | 0.308 | 0.606 |
| all_roleplay | 0.73 [0.70, 0.76] | 0.15 (n=1606) | – | – |

| Axis (all prompts of this subset) | separation (folded AUC) |
|---|---|
| harmful vs benign | 0.74 [0.71, 0.77] |
| role-play vs plain | 0.53 [0.50, 0.56] |
| role-play vs plain, harmful only | 0.80 [0.74, 0.85] |
| role-play vs plain, benign only | 0.72 [0.67, 0.76] |

| Benign role-play vs benign plain (wrapper effect) | AUC |
|---|---|
| neutral_persona | 0.72 [0.67, 0.76] |
| neutral_fiction | 0.69 [0.64, 0.74] |
| jb_persona | 0.76 [0.71, 0.81] |
| all_roleplay | 0.72 [0.67, 0.76] |

| Harmful plain vs benign role-play | AUC |
|---|---|
| neutral_persona | 0.90 [0.85, 0.94] |
| neutral_fiction | 0.90 [0.86, 0.95] |
| jb_persona | 0.88 [0.83, 0.93] |
