def add_value(dict_obj, key, value):
    if key not in dict_obj or dict_obj.get(key)>value:
        dict_obj[key] = value


phone_details = {   'Mathew': 212323,
                    'Ritika': 334455,
                     'John'  : 35 }
add_value(phone_details,'John',34.5)
print(phone_details)
    
