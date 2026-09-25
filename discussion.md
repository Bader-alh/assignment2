## Problem 1

### B
If I only had access to the priors, I would likely guess that Johnson authored an unlabeled paper.

There are 36 Kennedy documents and 66 Johnson documents in the labeled datasets. therefore, the prior probabilities can be calculated as:

$$ p(\text{Kennedy}) = \frac{36}{102} \approx 0.353 $$

$$ p(\text{Johnson}) = \frac{66}{102} \approx 0.647 $$

Since $p(\text{Johnson}) > p(\text{Kennedy})$, Johnson has a larger prior probability. therefore Given no information from the words in unlabeled datasets, Johnson would be the better prediction.

### C

After organizing the speeches into a Dataframe, I compared the lengths of the documents by author. kennedy has 36 speeches with an avergae length of around 2879 words, while Johnson has 66 speeches with an average length of 3620 words.

This tells me that Johnson's speeches are geenrally longer on average in this dataset. Johnson also had the longest individual speech, at 14214 words compared to Kennedy's max at 7650 words.

Moreover, I looked at the most common words used by each author. I first removed common stopwords (did so in terminal not saved code). Kennedy's most frequent words are "world", "new", "country", and "free". while Johnson's most frequent words are: "president", "people", "think", and "world" (Johnson used "Mr" more than world but i made the executive decision to ignore that). this difference shows the different topics and vocabulary in their speeches, which can help us in authorship classificaiton.

### E

the prior probability estimates are:
$$ P(\text{Kennedy}) = 0.3529 $$

$$ P(\text{Johnson}) = 0.6471 $$

the likelihood matrix has shape: $$(2,24390)$$

the two rows correspond to the authors and the columns correspond to the words in the vocabulary. When smoothening parameter $\alpha$ is increased, the likelihood estimate gets less extreeme. In the results, increasing $\alpha$ raised the minimum likelihood and lowered max likelihood. this means more probability is assigned to words with low or zero counts, while frequent words hold less weight. Lidstone smoothing also prevented unseen words from recieving a 0 probability.

### F

The predicted author for the unlabeled speeches are:

- `speech_10_johnson.txt` → Johnson
- `speech_11_kennedy.txt` → Kennedy
- `speech_12_kennedy.txt` → Kennedy
- `speech_13_kennedy.txt` → Kennedy
- `speech_14_kennedy.txt` → Kennedy
- `speech_5_johnson.txt` → Johnson
- `speech_6_johnson.txt` → Kennedy
- `speech_7_johnson.txt` → Johnson
- `speech_8_johnson.txt` → Kennedy
- `speech_9_kennedy.txt` → Kennedy

## Problem 2

### A

the scikit-learn naive bayes classifier produced predictions that were very simmilar to the prediction from the previous naive bayes implementation. the two classifiers agree on 9/10 of unlabeled speeches.

the only difference for one speech that the previous implementation predicted as kennedy, while the scikit-learn implementation predicted as Johnson. Overall, Scikit-learn correcttly classified 9 out of 10 speeches while the previous model correctly classified 8 of 10.

## Problem 3

### A

for my implementation of naive bayes, accuracy was:

$$ 0.80 $$

and f-1 score was:

$$ 0.75 $$

for scikit-learn implementation, accuracy was:

$$ 0.90 $$

f-1 was:

$$ 0.8889 $$

The Scikit-learn implementaion performed slighlty better than the custom implementation on the test set, getting both higher accuracy and F-1 Score.

### B

The confusion matrix for my implementation was:

$$ 
\begin{bmatrix}
5 & 0 \\
2 & 3
\end{bmatrix}
$$

confusion matrix for scikit-learn implementation was:
$$ 
\begin{bmatrix}
5 & 0 \\
1 & 4
\end{bmatrix}
$$

both classifiers correctly classified all five Kennedy speeches. However, my implementation misclassified two Johnson speeches as Kennedy, while Scikit-learn classifier misclassified only one Johnson speech as Kennedy.

This shows that scikit-learn performed better on the Johnson speeches, while both performed equally good on the Kennedy speeches. although it is important to know that we cannot make this assumption due to the little data for training.