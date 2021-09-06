import yaml

if __name__ == '__main__':

    stream = open("config.yaml", 'r')
    dictionary = yaml.load(stream)
    for key, value in dictionary.items():
        print (key + " : " + str(value))
