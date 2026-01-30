import sys
try:
    print("🧪 Testing API...") 
    from api.main import app 
    print("✅ API imported successfully")
    
    if hasattr(app, 'routes'):
        print(f"✅ API has {len(app.routes)} routes")
    else:
        print("❌ API routes not found")
        sys.exit(1)
    
    print("✅ API validation passed")
except Exception as e:
    print(f"❌ API validation failed: {e}")
    sys.exit(1)
    
