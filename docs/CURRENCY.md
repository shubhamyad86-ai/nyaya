# Amounts in EUR, INR or DM

The model learned from the Statlog German Credit data: 1,000 loans from 1973-75 with amounts in
Deutsche Mark. People enter and read amounts in their own currency; Nyaya converts them to DM before
scoring and back for display. The score itself never depends on the currency you choose.

| Currency | Rate | Source |
|---|---|---|
| EUR | 1 € = 1.95583 DM | Fixed legal rate since 1 January 1999 (Council Regulation (EC) No 2866/98) |
| INR | 1 € = 106.27 ₹ | Market rate on 15 October 2026; change it in `config.yaml` |
| DM | 1 | The training data |

## Where you choose it

| Where | How |
|---|---|
| Credit check page (`/app`) | **Currency** menu next to the example applicants. The amount field, its limits and the codebook labels (`... < 100 DM` becomes `... < €51`) switch with it. The default is set by `currency.default` in `config.yaml`. |
| Upload page (`/batch`) | **Amounts in** menu: the currency of the file's `amount` column. The sample file is in DM, so the page switches to DM when you use it. Results show amounts in the file's currency. |
| API | `POST /batch/check?currency=EUR` (or `INR`, default `DM`). `POST /credit/score` stays in DM. |
| Command line | `python scripts/check_csv.py file.csv --currency EUR` |

Category values in a file keep the codebook spelling (`... < 100 DM`); only the display changes.
Euro and rupee amounts may have decimals; they are rounded to whole DM for the model.

## What the conversion does not do

- **No inflation adjustment.** 1,000 DM in 1974 bought far more than 511 € buys today, so the model's
  idea of a "large loan" is a 1970s one. A real lender would retrain on current loans in its own
  currency instead of converting.
- **No validation for India.** Converting rupees lets an Indian user try the tool, but the model has
  never seen an Indian applicant. The India rule pack says so on every report.

## Changing a rate

Edit `config.yaml`:

```yaml
currency:
  default: EUR
  per_eur:
    INR: 106.27        # new rate here
  inr_rate_date: "2026-10-15"
```

Restart the app (`Ctrl+C`, then `start_api.bat`). The pages show the rate and its date in the note
under the currency menu. `tests/test_currency.py` checks that a file in EUR or INR gets exactly the
same scores as the same file in DM.
