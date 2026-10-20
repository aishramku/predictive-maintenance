# Predictive Maintenance

## The Problem

Factory machines break down without warning. Every unexpected failure means downtime, higher repair costs, and wasted time and resources. The goal was to predict failures before they happen so maintenance can be planned ahead instead of scrambling after something breaks.

## The Data

About 100,000 records of machine info, sensor readings, and operating conditions, with 38 features. Around 35% of the records were failures.

## What I Did

- Cleaned the data and removed columns that would give away the answer, like repair cost and post-repair status, so the model couldn't cheat
- Combined sensor readings into summary stats (mean, max, min, standard deviation)
- Created new features, like combining machine age with days since last maintenance, and production load with temperature
- Tested Decision Tree, Random Forest, Gradient Boosting, a Neural Network, and a Voting Classifier
- Tuned the best models with GridSearchCV and RandomizedSearchCV
- Used PCA and K-Means to group machines by behavior

## Results

- The tuned Random Forest performed best, with an F1 score of 0.606 and a good balance between catching failures and avoiding false alarms
- Clustering split the machines into three clear risk groups: one with a 59.5% failure rate, one at 36.4%, and one at 23.1%

## Recommendation

Use the tuned Random Forest to flag machines likely to fail, and focus maintenance on the high-risk group first. That way the machines most likely to break get fixed before they go down, instead of spreading time evenly across everything.

Built in Python with Pandas, Scikit-learn, and Matplotlib in a Jupyter Notebook.
