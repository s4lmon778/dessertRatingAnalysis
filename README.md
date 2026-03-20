# Do Dessert Nutrients Predict a High Rating?

**By Cheuk Lam Chen and Tony Zhang** | DSC 80 at UCSD

## Introduction

This project explores a dataset of recipes and user ratings scraped from [food.com](https://food.com), originally used in a recommender systems research paper. The dataset contains two parts: `RAW_recipes.csv`, which describes recipes, and `RAW_interactions.csv`, which contains ratings submitted by users. After left-merging the two datasets, the combined dataset contains **234,429 rows**, where each row represents one user rating for a recipe. Ratings of 0 were replaced with `NaN` since a rating of 0 on food.com indicates that a user left a review without submitting a numerical score; it does not represent a true rating of zero. We then computed the average rating per recipe and added it back to the recipes dataset, resulting in **83,782 unique recipes** for analysis.

Since nutrition varies significantly across recipe types, we chose to focus specifically on **dessert recipes**, recipes that are tagged with `'desserts'`, `'desserts-easy'`, `'desserts-fruit'`, or `'frozen-desserts'`, giving us a focused dataset of **13,502 dessert recipes**.

Food ratings are not just about the taste; nutrition plays a growing role in how people perceive and choose recipes. Determining whether sugar, calories, or protein content is correlated with higher ratings could help recipe creators optimize not just for flavor, but for the greater foodies like us. Specifically, we explore whether the nutritional contents of a dessert recipe can predict whether it will be highly rated (≥ 4.0).

| Column | Description |
|---|---|
| `avg_rating` | Average rating for the recipe (1–5) from all user ratings |
| `calories` | Total calories (#) in the recipe (from `nutrition`) |
| `sugar` | Sugar (PDV — percentage of daily value) (from `nutrition`) |
| `protein` | Protein content (PDV — percentage of daily value) (from `nutrition`) |
| `total_fat` | Total fat (PDV — percentage of daily value) (from `nutrition`) |
| `n_ingredients` | Number of ingredients in the recipe |
| `n_steps` | Number of steps in the recipe |
| `minutes` | Time (minutes) to prepare the recipe |

## Data Cleaning and Exploratory Data Analysis

We performed the following steps to prepare the data for analysis:

1. Loaded and merged `RAW_recipes.csv` and `RAW_interactions.csv` using left merge on recipe ID, producing 234,429 rows.
2. Replaced ratings of 0 with `NaN`; on food.com, a rating of 0 means the user left a review without inputting a score, not a true zero rating.
3. Computed average rating per recipe and merged it back as `avg_rating`.
4. Parsed `tags` from a raw string into a Python list for filtering.
5. Filtered to dessert recipes by keeping only recipes tagged with `'desserts'`, `'desserts-easy'`, `'desserts-fruit'`, or `'frozen-desserts'`, yielding 13,502 recipes.
6. Parsed the `nutrition` column from a string into separate numeric columns: `calories`, `total_fat`, `sugar`, `sodium`, `protein`, `saturated_fat`, and `carbohydrates`.

All nutritional columns are complete with no missing values. The 634 missing values in `avg_rating` represent recipes that received no user ratings — this is expected and will be handled during modeling by dropping rows where the target variable is missing.

The distribution of average ratings is heavily left-skewed, with the majority of dessert recipes rated between 4.0 and 5.0. This suggests that users tend to rate recipes they enjoy, and very few desserts receive low ratings.

<iframe src="assets/rating_dist.html" width="800" height="600" frameborder="0"></iframe>

Sugar content in dessert recipes is right-skewed with a median of 92% PDV. The upper fence sits at 335% PDV, beyond which outliers exist — likely recipes where contributors entered whole-recipe nutrition rather than per-serving. The y-axis is capped at 500 PDV for readability; the full dataset is retained for analysis.

<iframe src="assets/sugar_dist.html" width="800" height="600" frameborder="0"></iframe>

Recipes tagged as both `desserts` and `desserts-easy` have an average rating of 5.0, although this is likely due to a small sample size. Frozen desserts have fewer ingredients (avg 5.94) and lower calories compared to standard desserts, while maintaining a relatively high average rating of 4.72.

## Assessment of Missingness

The missing values in the rating column are likely MNAR because the probability that a rating is missing may depend on the rating value itself. For example, users might be more likely to leave a rating when they have a strong opinion (very good or very bad) and less likely to rate recipes when they feel neutral about them. In this case, the missingness is directly related to the unobserved value of the rating, which fits the definition of Missing Not At Random (MNAR). Another plausible scenario is that users who disliked a recipe may choose not to rate it at all, meaning low ratings could be systematically missing. Since the missingness depends on the rating that would have been recorded, the mechanism would again be MNAR.

We tested whether the missingness of `avg_rating` depends on `sugar` (p=0.0586) — we fail to reject the null hypothesis at the 0.05 significance level. The missingness of `avg_rating` does **not** depend on sugar content.

We then tested `n_steps` (p=0.0002) — we reject the null hypothesis. The missingness of `avg_rating` **does** depend on `n_steps`. Recipes with more steps tend to have missing ratings more often, suggesting complex recipes receive fewer reviews. This is consistent with MAR (Missing At Random) with respect to `n_steps`.

## Hypothesis Testing

We tested whether high-sugar desserts (above the median sugar PDV) have a different proportion of highly rated recipes (≥ 4.0) than low-sugar desserts.

**Null Hypothesis (H₀):** The proportion of highly rated recipes is the same for high-sugar and low-sugar desserts. Any observed difference is due to random chance.

**Alternative Hypothesis (H₁):** High-sugar desserts have a different proportion of highly rated recipes compared to low-sugar desserts.

**Test Statistic:** Difference in proportion of highly rated recipes (avg_rating ≥ 4.0) between high-sugar and low-sugar groups. We use difference in proportions because our response variable is binary (highly rated or not), consistent with permutation testing.

**Significance Level:** 0.05

With an observed difference in proportions of -0.0253 and a p-value of < 0.0001, we reject the null hypothesis at the 0.05 significance level. This demonstrates that sugar content is associated with the likelihood of a dessert being highly rated. Low-sugar desserts appear to have a slightly higher proportion of highly rated recipes than high-sugar ones. However, we cannot conclude causation from this test. The difference in proportions is small (-0.025), meaning the practical effect is modest even if statistically significant. Other factors likely play a larger role in determining whether a dessert is highly rated.

<iframe src="assets/hypothesis_test.html" width="800" height="600" frameborder="0"></iframe>

## Framing a Prediction Problem

We frame this as a **binary classification** problem. Our goal is to predict whether a dessert recipe will be highly rated (avg_rating ≥ 4.0) based on its nutritional content and complexity, with information that is available at the time a recipe is submitted to food.com.

**Response variable:** `is_highly_rated` — 1 if avg_rating ≥ 4.0, 0 otherwise. We chose this over predicting the exact avg_rating because the ratings cluster heavily near 5.0, making regression difficult. A binary target is cleaner and more helpful.

**Features used — all known before any ratings are submitted:**
- `calories` — the recipe creator calculates this before posting
- `sugar` — extracted from nutrition info entered at submission
- `protein` — extracted from nutrition info entered at submission
- `n_ingredients` — the creator knows how many ingredients they used
- `n_steps` — the creator wrote the steps themselves

We purposely exclude `avg_rating` as a feature since it only exists after users have rated the recipe — using it would be data leakage.

**Evaluation metric:** F1-score. We use F1 over accuracy because the classes are imbalanced. The majority of desserts are highly rated (≥ 4.0), so a model that always predicts "highly rated" would achieve high accuracy but be useless. F1 balances precision and recall, making it fair for predictions.

## Baseline Model

The baseline logistic regression model received an accuracy of 0.4650 and an F1-score of 0.6041 on the test set. The low accuracy is expected given the severe class imbalance; 91% of desserts are highly rated, meaning a naive model would achieve 90% accuracy by always predicting "highly rated." By using `class_weight='balanced'`, the model is forced to learn patterns in both classes, resulting in a more honest F1-score of 0.60. This gives us a starting point: the final model should achieve a higher F1-score by using better features and a more powerful algorithm.

The model uses 5 quantitative features: `calories`, `sugar`, `protein`, `n_ingredients`, and `n_steps`. All features are standardized using `StandardScaler`. There are no nominal or ordinal features in this baseline.

## Final Model

We trained a Random Forest classifier that extends the baseline model by incorporating two additional engineered features. First, we applied a logarithmic transformation to the calories variable (`log_calories`) to address its right-skewed distribution, reducing the influence of extreme values. Second, we constructed a `sugar_per_calorie` feature to capture relative sweetness, allowing the model to compare recipes independently of portion size.

We tuned the model using 5-fold cross-validation with GridSearchCV, which selected `max_depth=7` and `n_estimators=200` as the optimal hyperparameters. Constraining the tree depth helps limit overfitting and improves generalization to unseen data.

The final model achieved an F1-score of 0.8595, representing an improvement of 0.2554 over the baseline score of 0.6041. The model produces more balanced predictions across classes, identifying both highly rated (2597) and not highly rated (620) recipes rather than favoring the majority class.

## Fairness Analysis

Our two groups for the fairness analysis are simple desserts vs elaborate desserts. Simple desserts are defined as recipes with `n_ingredients` at or below the median (≤ 8), and elaborate desserts are those with more than the median number of ingredients (> 8).

We chose these groups because recipe complexity directly relates to our prediction task — a model that works well for simple 5-ingredient recipes but fails on complex 20-ingredient ones would be unfair to elaborate recipe creators.

**Null Hypothesis:** The model is fair. Its F1-score for simple and elaborate desserts are roughly the same, and any differences are due to random chance.

**Alternative Hypothesis:** The model is unfair. Its F1-score for simple desserts is significantly different from its F1-score for elaborate desserts.

We performed a permutation test with 1000 trials at a significance level of 0.05. The F1-score for simple desserts was 0.8430 and for elaborate desserts was 0.8765, giving an observed difference of -0.0335. With a p-value of 0.0010, we reject the null hypothesis — the model performs slightly but significantly better on elaborate desserts, possibly because they have more nutritional variation that gives the model more signal to work with.

<iframe src="assets/fairness.html" width="800" height="600" frameborder="0"></iframe>
