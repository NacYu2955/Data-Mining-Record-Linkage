from preprocessing import Preprocessor

examples = [
    # Letter Case
    ("JAMES", "James"),
    # Special Character
    ("Emma", "Em-ma"), ("M & M L.L.C.", "MM LLC"),
    # More/Less Space
    ("Anna", "An na"),
    # More/Less Word
    ("John Edward Smith", "John Smith"), ("Lee", "Mr. Lee"),
    # Word Reorder
    ("Carlos Alfonzo Diaz", "Diaz Carlos Alfonzo"),
    # Word Truncation
    ("Technology", "Tech"),
    # Variant Spelling
    ("Abdul Rasheed", "Abd al-Rashid"), ("Sarah", "Sara"),
    # Typographical Error
    ("Center", "Cetner"),
    # Initials
    ("James Earl Smith", "J.E. Smith"), ("J. E. Smith", "JE Smith"), ("J E Smith", "J.E. Smith"),
    # Abbreviation
    ("ltd", "limited"), ("shpg", "shipping"), ("apt", "apartment"), ("No.", "number"), ("NE", "northeast"), ("rd", "road"), ("US", "USA"), ("USA", "United States of America"),
    # Semantic Matching
    ("Eagle Pharmaceuticals Inc.", "Eagle Drugs Co"),
    # Transliteration
    ("company", "compañía"), ("公司", "회사"),
    # Nickname
    ("William", "Billy"), ("Robert", "Bob"), ("Elizabeth", "Betty"),
    # Number
    ("4", "four"), ("IV", "4"), ("1st", "first"),
    # Similar Shape
    ("0", "O"), ("1", "I"), ("2", "Z"), ("5", "S"), ("6", "b"), ("8", "B"), ("13", "B"), ("m", "rn"), ("L", "|_"), ("W", "VV"),
]

pre = Preprocessor()

print("验证图片中所有变体类型的预处理效果：\n")
for a, b in examples:
    pa = pre.preprocess(a)
    pb = pre.preprocess(b)
    print(f"原始: '{a}' <-> '{b}'  ==>  预处理: '{pa}' <-> '{pb}'  {'✔' if pa == pb else '✗'}") 