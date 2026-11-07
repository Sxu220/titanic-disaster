# Titanic Disaster – Survival Prediction (Python & R in Docker)

This repository predicts which passengers survived the sinking of the Titanic (April 15, 1912)
using **logistic regression**. The same pipeline is implemented twice, each in its own Docker
container:

| Container | Language | Model | Script |
|-----------|----------|-------|--------|
| `src/python` | Python 3.11 | scikit-learn `LogisticRegression` | `titanic.py` |
| `src/r` | R 4.4.1 | base R `glm(family = binomial)` | `titanic.R` |

Expected results: training accuracy is about **83%** for Python and **80%** for R.

Each script loads the data, prints what it explores and every change it makes to the data,
trains the model, reports training accuracy, predicts on the test set, and reports test accuracy.
You only need to read the terminal output; you don't need to open the code.

---

## Repository layout

```
titanic-disaster/
├── .github/
│   └── CODEOWNERS          # optional: restricts who can approve changes
├── .gitignore              # keeps the dataset, venvs and OS files out of git
├── README.md
└── src/
    ├── data/               # put the Kaggle CSVs here (ignored by git, NOT in the repo)
    │   └── .gitkeep
    ├── python/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   └── titanic.py
    └── r/
        ├── Dockerfile
        ├── install_packages.R
        └── titanic.R
```

> The dataset is **not** included in this repository. You download it yourself in Step 2.

---

## Prerequisites

1. **Git** – <https://git-scm.com/downloads>
2. **Docker Desktop** (or Docker Engine on Linux) – <https://docs.docker.com/get-docker/>
   Make sure Docker is **running** before Step 3. Check with:
   ```bash
   docker --version
   ```
3. A free **Kaggle account** – <https://www.kaggle.com> (needed to download the data).

---

## Step 1 – Clone the repository

```bash
git clone https://github.com/Sxu220/titanic-disaster.git
cd titanic-disaster
```

All commands below are run from this top-level `titanic-disaster` folder.

---

## Step 2 – Download the data

1. Go to <https://www.kaggle.com/competitions/titanic/data>, sign in, and accept the competition rules
   (click **"Join Competition"** / **"I Understand and Accept"** if asked).
2. Click **"Download All"**. This downloads `titanic.zip`.
3. Unzip it and move these three files into `src/data/`:
   - `train.csv` (891 passengers, with the `Survived` label)
   - `test.csv` (418 passengers, without the label)
   - `gender_submission.csv` (Kaggle's sample submission)

   macOS / Linux example (if the zip is in your Downloads folder):
   ```bash
   unzip ~/Downloads/titanic.zip -d src/data/
   ```
   Windows (PowerShell):
   ```powershell
   Expand-Archive $HOME\Downloads\titanic.zip -DestinationPath src\data
   ```

   **Alternative – Kaggle CLI** (if you have an API token in `~/.kaggle/kaggle.json`):
   ```bash
   pip install kaggle
   kaggle competitions download -c titanic -p src/data
   unzip src/data/titanic.zip -d src/data/
   ```

4. Confirm the files are in place:
   ```bash
   ls src/data
   # gender_submission.csv  test.csv  train.csv
   ```

---

## Step 3 – Run the Python container

Build the image (installs the pinned packages in `requirements.txt`):

```bash
docker build -t titanic-python src/python
```

Run it, mounting the data folder into the container:

```bash
# macOS / Linux / Git Bash
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-python
```
```powershell
# Windows PowerShell
docker run --rm -v "${PWD}/src/data:/app/data" titanic-python
```
```bat
:: Windows Command Prompt
docker run --rm -v "%cd%/src/data:/app/data" titanic-python
```

---

## Step 4 – Run the R container

Build the image (runs `install_packages.R`; the first build takes a few minutes):

```bash
docker build -t titanic-r src/r
```

Run it the same way:

```bash
# macOS / Linux / Git Bash
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-r
```
```powershell
# Windows PowerShell
docker run --rm -v "${PWD}/src/data:/app/data" titanic-r
```

---

## What you will see

Both scripts print the same seven sections:

1. **Load training data** – shape, first rows, missing values per column.
2. **Explore** – survival rate overall and by sex, class and port of embarkation.
3. **Clean and engineer features** – each change is printed:
   - extract a `Title` (Mr, Mrs, Miss, Master, Rare) from `Name`
   - fill missing `Age` with the median age for that title
   - fill missing `Fare` with the median fare and missing `Embarked` with the most common port
   - add `FamilySize` (SibSp + Parch + 1) and `IsAlone`
   - encode `Sex`, `Embarked` and `Title` as indicator variables
   - drop `Cabin` (mostly missing), `Ticket`, `Name`, `PassengerId`

   All fill values are learned from the **training set only** and reused on the test set.
4. **Fit logistic regression** – model coefficients. The two models use different features:
   - **Python** uses all 14 engineered features: Pclass, Sex, Age, SibSp, Parch, Fare,
     FamilySize, IsAlone, Embarked, Title. scikit-learn's default L2 regularization keeps the
     overlapping features stable.
   - **R** uses a smaller, non-overlapping set: Pclass, Sex, Age, Fare, FamilySize, IsAlone,
     Embarked. `glm` has no regularization, so it leaves out SibSp/Parch (already counted in
     FamilySize) and Title (overlaps with Sex).
5. **Training accuracy.**
6. **Predict on test set** – predictions are written to `src/data/predictions_python.csv`
   or `src/data/predictions_r.csv`.
7. **Test accuracy** – see the note below.

### A note on test accuracy

Kaggle does **not** publish survival labels for `test.csv`. The scripts therefore compare their
test predictions with `gender_submission.csv`, Kaggle's sample baseline in which every woman
survives and every man dies. That number shows how closely the model agrees with the baseline,
not its true accuracy. To get the true test accuracy, upload `predictions_python.csv` or
`predictions_r.csv` at <https://www.kaggle.com/competitions/titanic/submit>.

If `gender_submission.csv` is missing, the scripts skip this step and say so.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ERROR: could not find train.csv at /app/data/train.csv` | The CSVs aren't in `src/data/`, or you ran `docker run` from a different folder. `cd` to the repo root and check `ls src/data`. |
| `Cannot connect to the Docker daemon` | Start Docker Desktop and try again. |
| Windows: volume mount is empty | In Docker Desktop → Settings → Resources → File Sharing, allow the drive containing the repo. |
| Apple Silicon (M1/M2/M3) build warnings | Both base images are multi-arch, so they should work. If not, add `--platform linux/amd64` to `docker build` and `docker run`. |

---

## Running without Docker (optional)

```bash
# Python
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r src/python/requirements.txt
DATA_DIR=src/data python src/python/titanic.py

# R (R 4.x installed locally)
Rscript src/r/install_packages.R
DATA_DIR=src/data Rscript src/r/titanic.R
```
