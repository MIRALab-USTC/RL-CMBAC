import copy
def get_subclass(base_class, class_name):
    for c in base_class.__subclasses__():
        if c.__name__ == class_name:
            return c
    for c in base_class.__subclasses__():
        temp_c = get_subclass(c, class_name)
        if temp_c is not None:
            return temp_c
    return None

def _plural(word):
    if word.endswith('y'):
        return word[:-1] + 'ies'
    elif word[-1] in 'sx' or word[-2:] in ['sh', 'ch']:
        return word + 'es'
    elif word.endswith('an'):
        return word[:-2] + 'en'
    else:
        return word + 's'

def _under_score_to_camel(word):
    parts = word.split('_')
    parts = [p[0].upper() + p[1:] for p in parts]
    return ''.join(parts)

def get_item_class(item_type, item_class_name):
    path = 'mbrl.'+_plural(item_type)
    item_type = item_type.split(".")[-1]
    base_class_name = _under_score_to_camel(item_type)
    item_class_name = _under_score_to_camel(item_class_name)
    import importlib
    module = importlib.import_module(path)
    base_class = getattr(module, base_class_name)
    item_class = get_subclass(base_class, item_class_name)
    if item_class is None:
        raise RuntimeError("There is no %s class corresponding to %s"%(item_type,item_class_name))
    return item_class

def get_item(item_type, item_class_name, kwargs):
    item_class = get_item_class(item_type, item_class_name)
    item = item_class(**kwargs)
    return item

def _visit_all_items(config): 
    for item_type, param in config.items():
        if item_type == 'experiment':
            continue
        if isinstance(param, list):
            for p in param:
                item_name = p['name']
                item_kwargs = p.get('kwargs', {})
                yield item_name, item_type, p['class'], item_kwargs
        else:
            item_name = param.get('name', item_type)
            item_kwargs = param.get('kwargs', {})
            yield item_name, item_type, param['class'], item_kwargs
            
def get_dict_of_items_from_config(config):
    item_dict = {}
    for item_name, item_type, _, _ in _visit_all_items(config):
        item_dict[item_name] = None
    total_instance = 0

    def replace_kwargs(kwargs):
        ready = True
        for k, v in kwargs.items():
            if isinstance(v, str) and v[0] == '$':
                assert v[1:] in item_dict, "Please check your config file. There is no item corresponding to %s"%v
                item = item_dict[v[1:]]
                if item is not None:
                    kwargs[k] = item
                else:
                    ready = False
        return ready 

    while total_instance < len(item_dict):
        for item_name, item_type, item_class_name, item_kwargs in _visit_all_items(config):
            if item_dict[item_name] is not None:
                continue
            if replace_kwargs(item_kwargs):
                item = get_item(item_type, item_class_name, item_kwargs)
                item_dict[item_name] = item
                total_instance += 1
    print(item_dict)
    return item_dict
