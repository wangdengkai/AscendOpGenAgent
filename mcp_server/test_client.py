#!/usr/bin/env python3
"""
MCP Client 测试脚本

用于验证 mcp_client.py 的基本功能是否正常
"""

import sys
from pathlib import Path

# 添加项目根目录
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_import():
    """测试导入"""
    print("🧪 Test 1: Import mcp_client module")
    try:
        from mcp_server.mcp_client import AscendCMCPClient
        print("✅ Import successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_client_creation():
    """测试客户端创建"""
    print("\n🧪 Test 2: Create client instance")
    try:
        from mcp_server.mcp_client import AscendCMCPClient
        client = AscendCMCPClient(server_url="http://localhost:9002")
        print(f"✅ Client created successfully")
        print(f"   Server URL: {client.server_url}")
        client.close()
        return True
    except Exception as e:
        print(f"❌ Client creation failed: {e}")
        return False

def test_context_manager():
    """测试上下文管理器"""
    print("\n🧪 Test 3: Context manager")
    try:
        from mcp_server.mcp_client import AscendCMCPClient
        with AscendCMCPClient(server_url="http://localhost:9002") as client:
            print(f"✅ Context manager works")
            print(f"   Server URL: {client.server_url}")
        return True
    except Exception as e:
        print(f"❌ Context manager failed: {e}")
        return False

def test_api_methods_exist():
    """测试 API 方法是否存在"""
    print("\n🧪 Test 4: Check API methods")
    try:
        from mcp_server.mcp_client import AscendCMCPClient
        
        required_methods = [
            'upload_task',
            'build_kernel',
            'verify_accuracy',
            'benchmark_performance',
            'get_task_status',
            'execute_command',
            'download_results',
            'full_evaluation',
            'close'
        ]
        
        client = AscendCMCPClient()
        missing = []
        
        for method in required_methods:
            if not hasattr(client, method):
                missing.append(method)
        
        client.close()
        
        if missing:
            print(f"❌ Missing methods: {missing}")
            return False
        else:
            print(f"✅ All {len(required_methods)} methods exist")
            for method in required_methods:
                print(f"   ✓ {method}")
            return True
            
    except Exception as e:
        print(f"❌ Method check failed: {e}")
        return False

def test_argument_parser():
    """测试命令行参数解析"""
    print("\n🧪 Test 5: Argument parser")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "mcp_client.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and "full-eval" in result.stdout:
            print("✅ Argument parser works")
            print(f"   Available commands found in help")
            return True
        else:
            print(f"❌ Argument parser failed")
            print(f"   Return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Argument parser test failed: {e}")
        return False

def test_file_structure():
    """测试文件结构"""
    print("\n🧪 Test 6: File structure")
    try:
        required_files = [
            SCRIPT_DIR / "mcp_client.py",
            SCRIPT_DIR / "mcp_client.bat",
            SCRIPT_DIR / "mcp_client.sh",
            SCRIPT_DIR / "README_CLIENT.md",
            SCRIPT_DIR / "CLIENT_USAGE.md",
            SCRIPT_DIR / "EXAMPLES.md",
        ]
        
        missing = []
        for file in required_files:
            if not file.exists():
                missing.append(file.name)
        
        if missing:
            print(f"❌ Missing files: {missing}")
            return False
        else:
            print(f"✅ All {len(required_files)} files exist")
            for file in required_files:
                print(f"   ✓ {file.name}")
            return True
            
    except Exception as e:
        print(f"❌ File structure check failed: {e}")
        return False

def main():
    """运行所有测试"""
    print("=" * 80)
    print("MCP Client Test Suite")
    print("=" * 80)
    
    tests = [
        test_import,
        test_client_creation,
        test_context_manager,
        test_api_methods_exist,
        test_argument_parser,
        test_file_structure,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Test crashed: {e}")
            results.append(False)
    
    # 总结
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
