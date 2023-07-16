import configparser
import io


def parse():
    config = configparser.ConfigParser()
    config.read("utils/config.ini")
    return config
