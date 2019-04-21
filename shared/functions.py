
import os


def directory_contents(directory):
    return [f"{directory}/{x}" for x in os.listdir(directory)]
