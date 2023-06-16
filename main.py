from model import Model
import privateModule

if __name__ == '__main__':
    myModel = Model()

    myModel.Learn(privateModule.pathIn1, privateModule.pathIn2, 5, 5)

    myModel.GenerateImage(privateModule.pathIn1, privateModule.pathOut)

    del myModel
