To train the classifier on files with a file extension:
./preprocess.py --name=filename

To run the classifier on the reserved test data:
./classifier.py

To run the classifier on the reserved test data and compare to
the test answers:
./classifier.py --Ytest=Yte.npy

To run the classifier on specified test data:
./classifier.py --from_text=(filename or folder name)

To run the classifier on specified test data and compare
to specified answers (must be saved np.array format):
./classifier.py --from_text=(filename or folder name) --Ytest=answer_file_name.npy


The preprocessor can use TfidfVectorizer with an altered tokenizer
or with the -cv argument will use the custom CodeVectorizer

The classifier can be either a Decision Tree (default) or 
you can specify Multinomial Naive Bayes by passing
-bayes to ./classifier.py
