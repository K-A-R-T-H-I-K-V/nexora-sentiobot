# mock_db.py

from datetime import datetime, timedelta

# ==============================================================================
#  SYNCHRONIZED PRODUCT DATABASE
#  This data is based on the product manuals you provided.
#  Current Date for testing: September 30, 2025
# ==============================================================================

PRODUCTS = {
    # --- Nexora Thermostat Pro (2-year / 24-month warranty) ---
    "SN-NTS-PRO-XYZ987": {
        "product_name": "Nexora Thermostat Pro",
        "purchase_date": datetime.strptime("2023-08-15", "%Y-%m-%d"),
        # This warranty is EXPIRED as of Aug 2025.
        "warranty_months": 24
    },
    "SN-NTS-PRO-ABC123": {
        "product_name": "Nexora Thermostat Pro",
        "purchase_date": datetime.strptime("2024-11-01", "%Y-%m-%d"),
        # This warranty is ACTIVE.
        "warranty_months": 24
    },

    # --- LumiGlow Smart Light (1-year / 12-month warranty) ---
    "SN-NLRGB-LMO456": {
        "product_name": "LumiGlow Smart Light",
        "purchase_date": datetime.strptime("2025-05-20", "%Y-%m-%d"),
        # This warranty is ACTIVE.
        "warranty_months": 12
    },

    # --- SecureSphere 360 Camera (1-year / 12-month warranty) ---
    "SN-NCS360-CAM789": {
        "product_name": "SecureSphere 360 Camera",
        "purchase_date": datetime.strptime("2024-02-10", "%Y-%m-%d"),
        # This warranty is EXPIRED as of Feb 2025.
        "warranty_months": 12
    }
}

# ==============================================================================
#  SYNCHRONIZED ORDER DATABASE
#  These orders now contain the real products from your documentation.
# ==============================================================================

ORDERS = {
    "NX-2025-301": {
        "status": "Shipped",
        "shipped_on": "2025-09-28",
        "items": ["SecureSphere 360 Camera", "LumiGlow Smart Light"]
    },
    "NX-2025-302": {
        "status": "Processing",
        "shipped_on": None,
        "items": ["Nexora Thermostat Pro"]
    },
    "NX-2025-303": {
        "status": "Delivered",
        "shipped_on": "2025-09-22",
        "items": ["LumiGlow Smart Light"]
    }
}

USERS = {
    "alice": {
        "password": "password123", # Add this
        "name": "Alice",
        "owned_products": ["Nexora Thermostat Pro", "LumiGlow Bulb"]
    },
    "bob": {
        "password": "password456", # Add this
        "name": "Bob",
        "owned_products": ["Sentio Smart Hub"]
    },
    "guest": {
        "password": "", # Guest has no password
        "name": "Guest",
        "owned_products": []
    }
}