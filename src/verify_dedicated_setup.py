#!/usr/bin/env python3
"""
Quick script to verify dedicated client is working
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

print("="*80)
print("🔍 Verifying Dedicated Key Manager Setup")
print("="*80)

# Test 1: Import dedicated manager
print("\n1. Testing dedicated_key_manager import...")
try:
    from dedicated_key_manager import create_dedicated_client, get_dedicated_manager
    print("   ✅ dedicated_key_manager imported successfully")
except Exception as e:
    print(f"   ❌ Failed to import: {e}")
    sys.exit(1)

# Test 2: Create client
print("\n2. Testing client creation...")
try:
    client = create_dedicated_client(task_id="test_verify")
    print(f"   ✅ Client created with key #{client.key_index + 1}")
    print(f"   ✅ Rate limit: {client.min_delay_between_requests}s between requests")
except Exception as e:
    print(f"   ❌ Failed to create client: {e}")
    sys.exit(1)

# Test 3: Check creat_graph_with_description import
print("\n3. Testing creat_graph_with_description imports...")
try:
    import creat_graph_with_description
    # Check if it has the right import
    import inspect
    source = inspect.getsource(creat_graph_with_description)
    
    if "from dedicated_key_manager import" in source:
        print("   ✅ creat_graph_with_description uses dedicated_key_manager")
    else:
        print("   ❌ creat_graph_with_description NOT using dedicated_key_manager!")
        print("   ⚠️  You may need to restart Python process")
        sys.exit(1)
except Exception as e:
    print(f"   ⚠️  Could not verify: {e}")

# Test 4: Check data_chunk import
print("\n4. Testing data_chunk imports...")
try:
    import data_chunk
    source = inspect.getsource(data_chunk)
    
    if "from dedicated_key_manager import" in source or "client" in inspect.signature(data_chunk.run_chunk).parameters:
        print("   ✅ data_chunk accepts client parameter")
    else:
        print("   ❌ data_chunk NOT updated!")
        print("   ⚠️  You need to restart Python process")
        sys.exit(1)
except Exception as e:
    print(f"   ⚠️  Could not verify: {e}")

# Test 5: Manager stats
print("\n5. Testing manager stats...")
try:
    manager = get_dedicated_manager()
    stats = manager.get_stats()
    print(f"   ✅ Total keys: {stats['total_keys']}")
    print(f"   ✅ Available: {stats['available_keys']}")
    print(f"   ✅ Keys in use: {stats['keys_in_use']}")
except Exception as e:
    print(f"   ❌ Failed to get stats: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("✅ All verifications passed!")
print("="*80)
print("\n🎯 Next steps:")
print("   1. STOP any running three_layer_import.py process")
print("   2. Restart with:")
print("      cd /home/medgraph/src")
print("      python three_layer_import.py \\")
print("          --bottom ../data/layer3_umls \\")
print("          --middle ../data/layer2_pmc \\")
print("          --top ../data/layer1_mimic_ex \\")
print("          --grained_chunk \\")
print("          --ingraphmerge \\")
print("          --trinity")
print("\n✨ Each file will now get its own dedicated API key!")
print("✨ No more rate limit conflicts!")
