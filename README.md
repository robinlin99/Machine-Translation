## NLP Quick Task - Machine Translation 
This is my solution to the quick task created by Professor Artzi as part of the application for Independent Study with the LIL Lab. 

## Summary
For this problem, I used two different models to tackle the binary classification task: Support Vector Machine (SVM) and Logistic Regression. For data pre-processing, I first extracted the training and testing data from the **.txt** files. Each training/testing sample has the following properties:
- Source (Chinese)
- Reference (Professional Translation)
- Candidate (Candidate Translation)
- BLEU (Bilingual Evaluation Understudy) Score
- Label (Ground Truth)

For training both models, I used two features as input: BLEU Score and Cosine Similarity. This is based on my intuition on how humans can approach this binary classification task: we are likely to categorize the candidate based on how "natural" or "similar" it is to natural human language. The BLEU Score and Cosine Similarity are two different measures that capture this idea of "similarity" to human language. 

After training both models, I evaluated their respective classification accuracies on the testing data as well as their F1 Scores. I achieved the following results:
- SVM: Accuracy = 74.14%, F1 = 0.7169811320754716
- Logistic Regression: Accuracy = 75.29%, F1 = 0.7361963190184049

Both models seem to perform equally well, with the logistic regressor performing a little bit better.


## Packages/Libraries Used
- [Sklearn](https://scikit-learn.org/stable/)
- [Numpy](https://numpy.org/)
- [NLTK](https://www.nltk.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

## File Structure

The runnable python code in this project are stored as follows:

```
.
├── extract.py
├── logistic_regression.py
├── svm.py
├── train.txt
└── test.txt
```

## Code Samples 

## Credits
The original problem proposed in this short task was created by Professor Artzi. 

## License
MIT © [Robin Lin]()
