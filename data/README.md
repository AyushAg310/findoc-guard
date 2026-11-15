# Data

Only **synthetic or consented** documents are used in this project. Never commit real cheques,
account numbers or personal data.

| Folder | Contents | In git? |
| --- | --- | --- |
| `samples/` | A handful of small synthetic test images used by the README and tests | Yes |
| `generators/` | Scripts that create synthetic cheques, GST invoices and fraud variants | Yes |
| `raw/` | Downloaded public datasets (CORD, SROIE, Doc3D, DocUNet, IDRBT) | No |
| `synthetic/` | Generated datasets | No |
| `processed/` | Pipeline outputs used for evaluation | No |

Public datasets are downloaded by scripts, not stored in the repo. Check each dataset's licence
before use.
