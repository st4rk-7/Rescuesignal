# RescueSignal: step-by-step plan (one small step at a time)

## How to learn with this guide: concepts first, code second
Our goal is to **understand machine learning**: the process, how things work and why each step exists. Learning Python syntax is not the goal. At the viva you must explain *what* each step does and *why*, not type code from memory.

For every step:
1. **Concept.** Learn the idea in plain English first: what it is, and what problem it solves.
2. **Why it matters here.** Connect it to RescueSignal (unseen disasters, rare labels, leakage).
3. **Run the code.** AI help or vibe coding is fine. You don't need to explain every line, but you must know what the code *does* and what its output *means*.
4. **Read the result.** Look at the numbers or charts and say what they tell you.
5. **Say it.** Answer the step's "Say it" question out loud, in your own words, without notes. If you can't, the step isn't finished.

You must still know what the tool names mean (Pipeline, TF-IDF, GroupKFold, Macro-F1), because the examiners will use them.

## Rules for every step
1. **One step at a time.** Don't start the next step until you can answer its "Say it" question.
2. **Answer open questions, not just multiple choice.** A rough sentence in your own words beats skipping. The viva only asks open questions.
3. **Never invent numbers.** Every number in the report must come from code we actually ran.
4. **Save your work.** Commit after each step (message like `step 1.3 load all events`), and once a week write in the logbook who did what.

Tick the boxes as you go.

---

## Phase 0: Get ready
- [ ] **0.1 Install.** Python, VS Code, Python + Jupyter extensions, `.venv`, `pip install pandas scikit-learn matplotlib jupyter tabulate`.
  *Done when:* a notebook prints the scikit-learn version.
- [ ] **0.2 Project folder.** Create `data/ notebooks/ src/ results/`, make a private GitHub repo and invite your teammate.
- [ ] **0.3 Get the data.** Download HumAID **Set 1** from crisisnlp.qcri.org/humaid_dataset into `data/`, then unzip it.
  *Done when:* you can say how many folders/files there are and what one file looks like.
  *Say it:* "What is HumAID Set 1?"

## Phase 1: Understand the data (no machine learning yet)
- [ ] **1.1 Look at one file.** Load one event's file with pandas and print 10 rows.
  *Learn:* a table has rows (tweets) and columns (id, text, label).
  *Say it:* "What does one row mean?"
- [ ] **1.2 The labels.** Print the label names. For each one, write your own made-up example tweet.
  *Say it:* "What is the difference between 'requests or urgent needs' and 'rescue, volunteering or donation effort'?"
- [ ] **1.3 Put all events in one table.** Load every file and add an `event` column. The authors split each event 70/10/20, but we join those parts back together because we make our own split by event.
  *Done when:* you note the total number of rows. Expect **43,409** tweets and **10** labels. The proposal says 47,868 and 11, so record the difference in the logbook.
  *Say it:* "Why do we throw away HumAID's own train/test split?"
- [ ] **1.4 Count things.** Count tweets per label and per event, and draw a bar chart of each.
  *Learn:* class imbalance.
  *Say it:* "Which label is rarest? Why is that a problem?"
- [ ] **1.5 Event × label table.** Make a table with events as rows, labels as columns and counts inside.
  *Say it:* "Which events are missing some labels?"
- [ ] **1.6 Duplicate audit.** Check for empty texts, duplicate tweet IDs, exact and normalised duplicate texts (lowercase, no links/@names/RT/punctuation), and texts that appear in more than one event. Drop the extra copies and save the clean table.
  *Say it:* "Why do duplicates make a score look better than it really is?"

**Checkpoint 1:** explain the dataset in 2 minutes to your teammate.

## Phase 2: Turn text into numbers
- [ ] **2.1 Clean text.** Write `clean(text)`: lowercase it, replace links with URL, @names with USER and numbers with NUM, and keep "not"/"no". Test it on 5 tweets.
  *Say it:* "Why do we keep 'not'?"
- [ ] **2.2 Count words.** Use `CountVectorizer` on 4 tiny sentences you write yourself and print the matrix.
  *Learn:* bag of words, which means one column per word.
- [ ] **2.3 TF-IDF.** Run `TfidfVectorizer` on the same 4 sentences and compare it with 2.2.
  *Say it:* "Why does 'the' get a low score and 'collapsed' a high one?"
- [ ] **2.4 Bigrams.** Set `ngram_range=(1,2)`.
  *Say it:* "What extra does 'not safe' give compared with 'not' and 'safe' alone?"

**Checkpoint 2:** explain TF-IDF without using the formula.

## Phase 3: First model, the simple (cheating) way
- [ ] **3.1 Random split.** `train_test_split` with 80/20, stratified by label.
- [ ] **3.2 Logistic Regression.** Fit it on TF-IDF and predict the test tweets.
  *Say it:* "What did the model learn?" (One weight per word per label.)
- [ ] **3.3 Scores.** Print accuracy, then `classification_report`.
  *Learn:* precision, recall, F1, Macro-F1.
  *Say it:* "Why is Macro-F1 our main score and not accuracy?"
- [ ] **3.4 Confusion matrix.** Plot it.
  *Say it:* "Which two labels does the model mix up most?"
- [ ] **3.5 Pipeline.** Put the cleaning, TF-IDF and model inside one sklearn `Pipeline`.
  *Say it:* "What is leakage, and how does a Pipeline stop it?"

**Checkpoint 3:** write down this "in-event" score. HumAID reported about 0.71–0.73 weighted F1 for SVM, so yours should be close.

## Phase 4: The main idea, testing on disasters the model never saw
- [ ] **4.1 Choose the test events.** Pick 2 events of different hazard types that cover the most labels, write the choice in the logbook, and don't touch them again until Phase 7.
  *Say it:* "Why must this choice be fixed before training?"
- [ ] **4.2 Leave-one-event-out.** Using the remaining events, train on all but one and test on the one left out, then repeat for each event. Use `GroupKFold` / `LeaveOneGroupOut` with groups = event.
- [ ] **4.3 Compare.** Put the random-split Macro-F1 (Phase 3) next to the unseen-event Macro-F1 (mean ± SD). The difference is the **generalisation drop**.
  *Say it:* "Why is the score lower on a new disaster?"
- [ ] **4.4 Look inside the model.** Print the top 15 words for each label. Do you see place names or disaster names like "harvey" or "kerala"?
  *Say it:* "What does this show about random splits?"

**Checkpoint 4:** explain the whole project idea in 1 minute. This is the heart of the viva.

## Phase 5: Compare the five models
- [ ] **5.1 Add the models one at a time:** Linear SVM, Multinomial NB, Decision Tree, Random Forest. For each one, learn its one-line idea first, then run it with the same event folds.
- [ ] **5.2 Class weights.** Set `class_weight="balanced"` for the 4 models that support it.
  *Say it:* "What does this do for rare labels?"
- [ ] **5.3 Results table.** Make one table with 5 models × Macro-F1, accuracy, macro precision/recall and weighted F1, each as mean ± SD.
- [ ] **5.4 Small tuning.** Take the best 1–2 models and try a few values of `C` and `min_df`, judged on the validation folds only.
  *Say it:* "Why don't we tune on the test events?"

## Phase 6: Hand-made features and the ablation
- [ ] **6.1 Five features first.** Hashtag count, URL count, question marks, "need/help" word count and "dead/injured" word count. Then build up to all 25 from Table 3 of the proposal.
- [ ] **6.2 Scaling.** Add `MinMaxScaler` inside the Pipeline.
  *Say it:* "Why min–max and not a standard scaler for Naive Bayes?"
- [ ] **6.3 Combine.** Use `ColumnTransformer` to put TF-IDF and the features side by side.
- [ ] **6.4 Ablation.** Run (A) TF-IDF only, (B) features only and (C) both, for each model.
  *Say it:* "Did the features help? For which labels?"

## Phase 7: The final test (run once)
- [ ] **7.1 Freeze everything.** Write the final settings in the logbook, train on all development events and test once on the 2 test events.
- [ ] **7.2 Per-label results.** Make a confusion matrix for each test event.
- [ ] **7.3 Error analysis.** Read 20 wrong predictions and write why each went wrong (sarcasm, two labels in one tweet, a new word, and so on).
- [ ] **7.4 (Optional) Vocabulary shift.** Find what % of the test event's words never appear in training, and compare it with its score.

## Phase 8: Report, slides, viva
- [ ] **8.1** Move the working notebook code into clean files in `src/` and write a README that explains how to run it.
- [ ] **8.2** Write the report: data, method, results tables, discussion, limitations.
- [ ] **8.3** Make the slides: problem, idea, data, method, results, what we learned.
- [ ] **8.4** Viva practice: answer every "Say it" question in this file without notes.
