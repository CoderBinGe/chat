# coding:utf-8
# from configparser import SafeConfigParser

# def get_config(config_file='seq2seq.ini'):
#     parser = SafeConfigParser()
#     parser.read(config_file)
#     # get the ints, floats and strings
#     _conf_ints = [ (key, int(value)) for key,value in parser.items('ints') ]
#     #_conf_floats = [ (key, float(value)) for key,value in parser.items('floats') ]
#     _conf_strings = [ (key, str(value)) for key,value in parser.items('strings') ]
#     return dict(_conf_ints  + _conf_strings)


import configparser  # 读取配置文件的包


def get_config(config_file='seq2seq.ini'):
    parser = configparser.ConfigParser()
    parser.read(config_file, encoding="utf-8")
    # 获取整参数，按照key-value的形式保存
    _conf_ints = [(key, int(value)) for key, value in parser.items('ints')]
    # 获取浮点型参数，按照key-value的形式保存
    # _conf_floats = [ (key, float(value)) for key,value in parser.items('floats') ]
    # 获取字符型参数，按照key-value的形式保存
    _conf_strings = [(key, str(value)) for key, value in parser.items('strings')]
    return dict(_conf_ints + _conf_strings)
