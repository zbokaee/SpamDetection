from sklearn import svm
import classification_Email
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import os

class DetecteurDeSpamEmail():
    
    def __init__(self, meilleur_clf):
        self.meilleur_clf = meilleur_clf
        self.X, self.y = classification_Email.creationBase("detectionl/train")
        self.action()
        self.enregistre()

    def action(self):
        text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                             ('clf', self.meilleur_clf)])
        text_clf.fit(self.X, self.y)

        X_test = []
        self.filenames_test = []
        for fichier in os.listdir('detection/test'):
            if not(os.path.isdir(fichier)) and fichier != ".DS_Store":
                self.filenames_test += [fichier]
                with open("detection/test/" + fichier, 'rb') as f:
                    email = f.read()
                    email = classification_Email.nettoieMail(email)
                    X_test += [email]
        self.y_test = text_clf.predict(X_test)

    def enregistre(self):
     
        fichier_sortie = open("resultats_TEST_Condaminet.txt", "w")
        for i in range(len(self.y_test)):
            y = self.y_test[i]
            filename = self.filenames_test[i]
            fichier_sortie.write(filename + " " + str(y) + "\n")
        fichier_sortie.close()

if __name__ == '__main__':
    DetecteurDeSpamEmail(svm.SVC(C=1000, gamma=0.01, kernel='rbf'))
