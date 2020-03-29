def printof(text, fobj):
    """
    Prints to output as well as file.
    
    Args:
        text (str): Text to be printed
        fobj (file object): File object
    """
    print(text)
    print(text, file=fobj)
