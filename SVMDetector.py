__author__ = "ZahraBokaee (2287369)"

 
from sklearn.datasets import load_files
import ClassificationDeTexte


def nettoieMail(email):
    
    lignes_email = email.split(b"\n")
    i = 0
    while i < len(lignes_email) and lignes_email[i] != b'':
        i += 1
    email = b"\n".join(lignes_email[i + 1:])
    return email.decode('utf-8', 'ignore')

def creationBase(chemin_container):
    
    train = load_files(container_path=chemin_container, random_state=42)
    
    X = []
    y = []
    for i in range(len(train.data)):
        if ".DS_Store" not in train.filenames[i]:
            email = train.data[i]
            email = nettoieMail(email)
            X += [email]
            classe = train.target_names[train.target[i]]
            y += [int(classe == 'spam')]
    
    return X, y

if __name__ == '__main__':
    X, y = creationBase("Base_Email/train")
    text_clf = ClassificationDeTexte.ClassificationDeTexte("Email", X, y, 0.33)
    text_clf.classification()
