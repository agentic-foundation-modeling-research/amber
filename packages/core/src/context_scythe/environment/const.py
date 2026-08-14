MUTABLE_SITES = [
    "shopping", "shopping_admin", "reddit", "gitlab"
]

FIXED_SITES = [
    "map", "wikipedia"
]

SITE_URL_TEMPLATES = {
    "shopping": "http://{host}:{port}",
    "shopping_admin": "http://{host}:{port}/admin",
    "reddit": "http://{host}:{port}",
    "gitlab": "http://{host}:{port}",
    "map": "http://{host}:{port}",
    "wikipedia": "http://{host}:{port}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",
    "homepage": "http://{host}:{port}",
    "calculator": "http://{host}:{port}/calculator.html"
}

ACCOUNTS = {
    "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
    "gitlab": {"username": "byteblaze", "password": "hello1234"},
    "shopping": {
        "username": "emma.lopez@gmail.com",
        "password": "Password.123",
    },
    "shopping_admin": {"username": "admin", "password": "admin1234"},
    "shopping_site_admin": {"username": "admin", "password": "admin1234"},
}
