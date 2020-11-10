def get_class_id(class_str, classes):
    if class_str in classes:
        class_id = classes.index(class_str)
    else:
        if class_str == '':
            raise ValueError("No class name found")
        else:
            class_id = -1000
    return class_id


def get_class_name(class_id, classes):
    n_class = len(classes)
    if 0 <= class_id < n_class:
        class_name = classes[class_id]
    elif class_id == -1000:
        class_name = '__background'
    else:
        raise ValueError("Class ID is not defined")
    return class_name
