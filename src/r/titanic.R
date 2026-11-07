# Titanic survival prediction - R version.
#
# Loads train.csv / test.csv from the data folder, cleans and engineers features,
# fits a logistic regression (glm with binomial family), and reports training and
# test accuracy.
#
# The data folder defaults to /app/data (the Docker mount point). Override it with
# the DATA_DIR environment variable when running outside Docker.

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

data_dir    <- Sys.getenv("DATA_DIR", "/app/data")
train_path  <- file.path(data_dir, "train.csv")
test_path   <- file.path(data_dir, "test.csv")
labels_path <- file.path(data_dir, "gender_submission.csv")
output_path <- file.path(data_dir, "predictions_r.csv")

banner <- function(text) {
  cat("\n", strrep("=", 70), "\n", text, "\n", strrep("=", 70), "\n", sep = "")
}

load_csv <- function(path, name) {
  if (!file.exists(path)) {
    cat(sprintf("ERROR: could not find %s at %s.\n", name, path))
    cat("Download it from https://www.kaggle.com/competitions/titanic/data",
        "and place it in src/data/ (see README.md).\n")
    quit(status = 1)
  }
  df <- read_csv(path, show_col_types = FALSE)
  cat(sprintf("Loaded %s: %d rows x %d columns\n", name, nrow(df), ncol(df)))
  df
}

extract_title <- function(name) {
  title <- trimws(sub("\\..*$", "", sub("^[^,]*,", "", name)))
  title <- recode(title, Mlle = "Miss", Ms = "Miss", Mme = "Mrs")
  ifelse(title %in% c("Mr", "Mrs", "Miss", "Master"), title, "Rare")
}

title_levels    <- c("Mr", "Mrs", "Miss", "Master", "Rare")
embarked_levels <- c("C", "Q", "S")

prepare <- function(df, age_by_title, age_all, fare_median, embarked_mode, label) {
  cat(sprintf("\n[%s] Preparing data\n", label))

  df <- df %>% mutate(Title = factor(extract_title(Name), levels = title_levels))
  counts <- table(df$Title)
  cat(sprintf("[%s] Added 'Title' from Name -> counts: %s\n", label,
              paste(names(counts), counts, sep = "=", collapse = ", ")))

  n <- sum(is.na(df$Age))
  fill <- age_by_title[as.character(df$Title)]
  fill[is.na(fill)] <- age_all
  df$Age <- ifelse(is.na(df$Age), fill, df$Age)
  cat(sprintf("[%s] Filled %d missing 'Age' values with the training median age for each title\n", label, n))

  n <- sum(is.na(df$Fare))
  df$Fare[is.na(df$Fare)] <- fare_median
  cat(sprintf("[%s] Filled %d missing 'Fare' values with training median %.2f\n", label, n, fare_median))

  n <- sum(is.na(df$Embarked))
  df$Embarked[is.na(df$Embarked)] <- embarked_mode
  cat(sprintf("[%s] Filled %d missing 'Embarked' values with training mode '%s'\n", label, n, embarked_mode))

  df <- df %>%
    mutate(
      FamilySize = SibSp + Parch + 1,
      IsAlone    = as.integer(FamilySize == 1),
      Sex        = factor(Sex, levels = c("female", "male")),
      Embarked   = factor(Embarked, levels = embarked_levels),
      Pclass     = as.numeric(Pclass)
    )
  cat(sprintf("[%s] Added 'FamilySize' (SibSp + Parch + 1) and 'IsAlone' (%.1f%% travelling alone)\n",
              label, 100 * mean(df$IsAlone)))
  cat(sprintf("[%s] Converted Sex, Embarked, Title to factors (glm one-hot encodes them)\n", label))
  cat(sprintf("[%s] Model features: Pclass, Sex, Age, Fare, FamilySize, IsAlone, Embarked\n", label))
  cat(sprintf("[%s] Not used in model: Cabin (mostly missing), Ticket, Name, PassengerId, SibSp/Parch (in FamilySize), Title (overlaps Sex; Title is still used to fill Age)\n", label))
  df
}

# The R model uses a smaller feature set than the Python one. SibSp and Parch are
# left out because FamilySize already equals SibSp + Parch + 1, and Title is left
# out because it overlaps almost completely with Sex (Mrs/Miss are always female).
# Unlike scikit-learn, glm has no regularization, so overlapping features make the
# coefficients unstable.
formula <- Survived ~ Pclass + Sex + Age + Fare + FamilySize + IsAlone + Embarked

# ---------------------------------------------------------------------------
banner("STEP 1: Load training data")
train <- load_csv(train_path, "train.csv")
cat("\nFirst 5 rows:\n")
print(as.data.frame(head(train, 5)))
cat("\nMissing values per column:\n")
na_counts <- colSums(is.na(train))
print(na_counts[na_counts > 0])

# ---------------------------------------------------------------------------
banner("STEP 2: Explore training data")
cat(sprintf("Overall survival rate: %.1f%%\n", 100 * mean(train$Survived)))
cat("\nSurvival rate by sex:\n")
print(as.data.frame(train %>% group_by(Sex) %>% summarise(rate = round(mean(Survived), 3))))
cat("\nSurvival rate by passenger class:\n")
print(as.data.frame(train %>% group_by(Pclass) %>% summarise(rate = round(mean(Survived), 3))))
cat("\nSurvival rate by port of embarkation:\n")
print(as.data.frame(train %>% filter(!is.na(Embarked)) %>%
                      group_by(Embarked) %>% summarise(rate = round(mean(Survived), 3))))

# ---------------------------------------------------------------------------
banner("STEP 3: Clean and engineer features")
train_titles  <- extract_title(train$Name)
age_by_title  <- tapply(train$Age, train_titles, median, na.rm = TRUE)
age_all       <- median(train$Age, na.rm = TRUE)
fare_median   <- median(train$Fare, na.rm = TRUE)
embarked_mode <- names(which.max(table(train$Embarked)))
cat("Imputation values learned from the training set only:\n")
cat("  Median age by title:", paste(names(age_by_title), round(age_by_title, 1), sep = "=", collapse = ", "), "\n")
cat(sprintf("  Median fare: %.2f   Most common port: %s\n", fare_median, embarked_mode))

train_p <- prepare(train, age_by_title, age_all, fare_median, embarked_mode, "train")

# ---------------------------------------------------------------------------
banner("STEP 4: Fit logistic regression")
model <- glm(formula, data = train_p, family = binomial())
cat(sprintf("Trained on %d passengers\n", nrow(train_p)))
cat("\nCoefficients (positive = more likely to survive):\n")
print(round(sort(coef(model), decreasing = TRUE), 3))

# ---------------------------------------------------------------------------
banner("STEP 5: Training accuracy")
train_pred <- as.integer(predict(model, train_p, type = "response") > 0.5)
train_acc  <- mean(train_pred == train_p$Survived)
cat(sprintf("Training accuracy: %.4f (%.1f%%)\n", train_acc, 100 * train_acc))

# ---------------------------------------------------------------------------
banner("STEP 6: Load test data and predict")
test   <- load_csv(test_path, "test.csv")
test_p <- prepare(test, age_by_title, age_all, fare_median, embarked_mode, "test")
test_pred <- as.integer(predict(model, test_p, type = "response") > 0.5)
out <- data.frame(PassengerId = test$PassengerId, Survived = test_pred)
write_csv(out, output_path)
cat(sprintf("\nPredicted survival for %d test passengers (%.1f%% predicted to survive)\n",
            nrow(out), 100 * mean(test_pred)))
cat(sprintf("Saved predictions to %s\n", output_path))

# ---------------------------------------------------------------------------
banner("STEP 7: Test accuracy")
truth <- NULL
if ("Survived" %in% names(test)) {
  truth  <- test$Survived
  source <- "Survived column in test.csv"
} else if (file.exists(labels_path)) {
  labels <- read_csv(labels_path, show_col_types = FALSE)
  truth  <- left_join(test["PassengerId"], labels, by = "PassengerId")$Survived
  source <- "gender_submission.csv"
}

if (is.null(truth)) {
  cat("Kaggle's test.csv has no Survived labels, and gender_submission.csv was not found,\n")
  cat("so test accuracy cannot be computed locally. Upload predictions_r.csv to Kaggle\n")
  cat("to get the official score.\n")
} else {
  test_acc <- mean(test_pred == truth)
  cat(sprintf("Test accuracy against %s: %.4f (%.1f%%)\n", source, test_acc, 100 * test_acc))
  if (source == "gender_submission.csv") {
    cat("NOTE: Kaggle hides the true test labels. gender_submission.csv is Kaggle's sample\n")
    cat("baseline (all women survive, all men die), so this number measures agreement with\n")
    cat("that baseline, not true accuracy. Submit predictions_r.csv to Kaggle for the\n")
    cat("real test score.\n")
  }
}

banner("DONE")
