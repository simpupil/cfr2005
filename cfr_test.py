#!/usr/bin/env python3
"""
CFR Installation Test Script

This script tests the basic functionality of the CFR (Climate Field Reconstruction) package
to verify that the installation was successful.

Usage:
    python cfr_test.py
"""

import sys
import numpy as np
import pandas as pd
import xarray as xr
import cfr

def test_imports():
    """Test importing main CFR modules"""
    print("="*60)
    print("CFR INSTALLATION TEST")
    print("="*60)
    print(f"CFR Version: {cfr.__version__}")
    print("\n1. Testing module imports...")
    
    try:
        from cfr import climate
        print("✓ climate module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import climate module: {e}")
        return False
    
    try:
        from cfr import proxy
        print("✓ proxy module imported successfully")  
    except ImportError as e:
        print(f"❌ Failed to import proxy module: {e}")
        return False
        
    try:
        from cfr import psm
        print("✓ psm module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import psm module: {e}")
        return False
        
    try:
        from cfr import utils
        print("✓ utils module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import utils module: {e}")
        return False
    
    return True

def test_climate_field():
    """Test ClimateField functionality"""
    print("\n2. Testing ClimateField functionality...")
    
    try:
        from cfr.climate import ClimateField
        
        # Create synthetic temperature data
        lat = np.arange(-90, 91, 10)
        lon = np.arange(-180, 181, 10) 
        time = pd.date_range('2000', '2005', freq='YS')
        
        # Create synthetic temperature field with realistic values
        temp_data = 15 + 10 * np.cos(np.radians(lat))[None, :, None] + np.random.randn(len(time), len(lat), len(lon)) * 2
        
        # Create DataArray
        da = xr.DataArray(
            temp_data,
            dims=['time', 'lat', 'lon'],
            coords={'time': time, 'lat': lat, 'lon': lon},
            name='temperature',
            attrs={'units': 'degrees_C', 'long_name': 'Surface Temperature'}
        )
        
        # Create ClimateField object
        cf = ClimateField(da)
        print(f"✓ Created ClimateField with shape: {cf.da.shape}")
        print(f"✓ Data dimensions: {list(cf.da.dims)}")
        print(f"✓ Temperature range: {cf.da.min().values:.2f}°C to {cf.da.max().values:.2f}°C")
        
        # Test some basic operations
        cf_copy = cf.copy()
        print("✓ ClimateField copy operation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing ClimateField: {e}")
        return False

def test_proxy_database():
    """Test ProxyDatabase functionality"""
    print("\n3. Testing ProxyDatabase functionality...")
    
    try:
        from cfr.proxy import ProxyDatabase
        print("✓ ProxyDatabase class available")
        
        # Test creating empty database
        pdb = ProxyDatabase()
        print("✓ Empty ProxyDatabase created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing ProxyDatabase: {e}")
        return False

def test_psm():
    """Test PSM (Proxy System Model) functionality"""
    print("\n4. Testing PSM functionality...")
    
    try:
        from cfr.psm import Linear
        print("✓ Linear PSM class available")
        
        # Create a simple linear PSM
        psm = Linear()
        print("✓ Linear PSM instance created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing PSM: {e}")
        return False

def test_cli():
    """Test command line interface"""
    print("\n5. Testing CLI functionality...")
    
    try:
        import subprocess
        result = subprocess.run(['cfr', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ CLI command works: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ CLI command failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing CLI: {e}")
        return False

def main():
    """Main test function"""
    tests = [
        test_imports,
        test_climate_field, 
        test_proxy_database,
        test_psm,
        test_cli
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! CFR installation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())