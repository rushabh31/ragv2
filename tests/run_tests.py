#!/usr/bin/env python3
"""
Comprehensive Test Runner for RAG System

This script provides a unified interface to run all tests in the RAG system,
organized by category with detailed reporting and filtering options.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

class TestRunner:
    """Comprehensive test runner for the RAG system."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_categories = {
            "unit": {
                "description": "Unit tests for individual components",
                "path": "tests/unit",
                "files": [
                    "test_embedding_models.py",
                    "test_generation_models.py"
                ]
            },
            "integration": {
                "description": "Integration tests for component interactions",
                "path": "tests/integration", 
                "files": [
                    "test_groq_and_sentence_transformers.py",
                    "test_langgraph_memory.py"
                ]
            },
            "performance": {
                "description": "Performance and parallel processing tests",
                "path": "tests/performance",
                "files": [
                    "test_parallel_performance.py",
                    "test_vision_parser_performance.py"
                ]
            },
            "system": {
                "description": "End-to-end system tests",
                "path": "tests/system",
                "files": [
                    "test_complete_system.py",
                    "test_factory_system.py"
                ]
            }
        }
        
    def print_banner(self):
        """Print test runner banner."""
        print("=" * 80)
        print("🧪 RAG System Test Runner")
        print("=" * 80)
        print()
        
    def list_tests(self):
        """List all available tests by category."""
        print("📋 Available Test Categories:")
        print()
        
        for category, info in self.test_categories.items():
            print(f"🔸 {category.upper()}")
            print(f"   Description: {info['description']}")
            print(f"   Path: {info['path']}")
            print(f"   Tests:")
            
            for test_file in info['files']:
                test_path = self.project_root / info['path'] / test_file
                if test_path.exists():
                    print(f"     ✅ {test_file}")
                else:
                    print(f"     ❌ {test_file} (missing)")
            print()
            
    def run_test_file(self, test_path: Path, verbose: bool = False) -> Dict:
        """Run a single test file and return results."""
        print(f"🔄 Running {test_path.name}...")
        
        start_time = time.time()
        
        try:
            cmd = [sys.executable, str(test_path)]
            if verbose:
                result = subprocess.run(cmd, cwd=self.project_root, 
                                      capture_output=False, text=True)
            else:
                result = subprocess.run(cmd, cwd=self.project_root,
                                      capture_output=True, text=True)
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ {test_path.name} - PASSED ({duration:.2f}s)")
                return {
                    "status": "PASSED",
                    "duration": duration,
                    "output": result.stdout if not verbose else "",
                    "error": ""
                }
            else:
                print(f"❌ {test_path.name} - FAILED ({duration:.2f}s)")
                if not verbose:
                    print(f"   Error: {result.stderr[:200]}...")
                return {
                    "status": "FAILED", 
                    "duration": duration,
                    "output": result.stdout if not verbose else "",
                    "error": result.stderr
                }
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"💥 {test_path.name} - ERROR ({duration:.2f}s)")
            print(f"   Exception: {str(e)}")
            return {
                "status": "ERROR",
                "duration": duration,
                "output": "",
                "error": str(e)
            }
            
    def run_category(self, category: str, verbose: bool = False) -> Dict:
        """Run all tests in a category."""
        if category not in self.test_categories:
            print(f"❌ Unknown category: {category}")
            return {"status": "ERROR", "results": []}
            
        info = self.test_categories[category]
        print(f"\n🚀 Running {category.upper()} tests...")
        print(f"📁 {info['description']}")
        print("-" * 60)
        
        results = []
        for test_file in info['files']:
            test_path = self.project_root / info['path'] / test_file
            
            if not test_path.exists():
                print(f"⚠️  {test_file} - SKIPPED (file not found)")
                results.append({
                    "file": test_file,
                    "status": "SKIPPED",
                    "duration": 0,
                    "output": "",
                    "error": "File not found"
                })
                continue
                
            result = self.run_test_file(test_path, verbose)
            result["file"] = test_file
            results.append(result)
            
        return {"status": "COMPLETED", "results": results}
        
    def run_all_tests(self, verbose: bool = False) -> Dict:
        """Run all tests across all categories."""
        print("\n🎯 Running ALL tests...")
        print("=" * 60)
        
        all_results = {}
        total_passed = 0
        total_failed = 0
        total_errors = 0
        total_skipped = 0
        total_duration = 0
        
        for category in self.test_categories.keys():
            category_result = self.run_category(category, verbose)
            all_results[category] = category_result
            
            for result in category_result["results"]:
                total_duration += result["duration"]
                if result["status"] == "PASSED":
                    total_passed += 1
                elif result["status"] == "FAILED":
                    total_failed += 1
                elif result["status"] == "ERROR":
                    total_errors += 1
                elif result["status"] == "SKIPPED":
                    total_skipped += 1
                    
        return {
            "results": all_results,
            "summary": {
                "total_passed": total_passed,
                "total_failed": total_failed,
                "total_errors": total_errors,
                "total_skipped": total_skipped,
                "total_duration": total_duration
            }
        }
        
    def print_summary(self, results: Dict):
        """Print test results summary."""
        summary = results["summary"]
        
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        
        print(f"✅ Passed:  {summary['total_passed']}")
        print(f"❌ Failed:  {summary['total_failed']}")
        print(f"💥 Errors:  {summary['total_errors']}")
        print(f"⚠️  Skipped: {summary['total_skipped']}")
        print(f"⏱️  Total Time: {summary['total_duration']:.2f}s")
        
        total_tests = (summary['total_passed'] + summary['total_failed'] + 
                      summary['total_errors'] + summary['total_skipped'])
        
        if total_tests > 0:
            success_rate = (summary['total_passed'] / total_tests) * 100
            print(f"📈 Success Rate: {success_rate:.1f}%")
            
        print()
        
        # Category breakdown
        print("📋 Results by Category:")
        for category, category_result in results["results"].items():
            passed = sum(1 for r in category_result["results"] if r["status"] == "PASSED")
            total = len(category_result["results"])
            print(f"   {category.upper()}: {passed}/{total} passed")
            
        print()
        
        # Overall status
        if summary['total_failed'] == 0 and summary['total_errors'] == 0:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("⚠️  SOME TESTS FAILED - Check output above for details")
            
    def check_environment(self):
        """Check if the environment is properly set up for testing."""
        print("🔍 Checking test environment...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 9):
            print(f"⚠️  Python {python_version.major}.{python_version.minor} detected. Python 3.9+ recommended.")
        else:
            print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
            
        # Check required environment variables
        required_vars = [
            "GROQ_API_KEY",
            "COIN_CONSUMER_ENDPOINT_URL",
            "COIN_CONSUMER_CLIENT_ID",
            "PROJECT_ID"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
                
        if missing_vars:
            print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
            print("   Some tests may fail without proper authentication.")
        else:
            print("✅ All required environment variables found")
            
        # Check test directories
        for category, info in self.test_categories.items():
            test_dir = self.project_root / info['path']
            if test_dir.exists():
                print(f"✅ {info['path']} directory exists")
            else:
                print(f"❌ {info['path']} directory missing")
                
        print()

def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="RAG System Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --list                    # List all available tests
  python run_tests.py --all                     # Run all tests
  python run_tests.py --category unit           # Run unit tests only
  python run_tests.py --category performance    # Run performance tests
  python run_tests.py --all --verbose           # Run all tests with verbose output
  python run_tests.py --check-env               # Check environment setup
        """
    )
    
    parser.add_argument("--list", action="store_true",
                       help="List all available tests")
    parser.add_argument("--all", action="store_true",
                       help="Run all tests")
    parser.add_argument("--category", choices=["unit", "integration", "performance", "system"],
                       help="Run tests from specific category")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output (show test output in real-time)")
    parser.add_argument("--check-env", action="store_true",
                       help="Check environment setup")
    parser.add_argument("--json", action="store_true",
                       help="Output results in JSON format")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if not any([args.list, args.all, args.category, args.check_env]):
        runner.print_banner()
        parser.print_help()
        return
        
    if not args.json:
        runner.print_banner()
        
    if args.check_env:
        runner.check_environment()
        
    if args.list:
        runner.list_tests()
        return
        
    if args.all:
        results = runner.run_all_tests(args.verbose)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            runner.print_summary(results)
            
    elif args.category:
        category_result = runner.run_category(args.category, args.verbose)
        if args.json:
            print(json.dumps(category_result, indent=2))
        else:
            # Print mini summary for single category
            results = category_result["results"]
            passed = sum(1 for r in results if r["status"] == "PASSED")
            failed = sum(1 for r in results if r["status"] == "FAILED")
            errors = sum(1 for r in results if r["status"] == "ERROR")
            skipped = sum(1 for r in results if r["status"] == "SKIPPED")
            
            print(f"\n📊 {args.category.upper()} Test Results:")
            print(f"✅ Passed: {passed}, ❌ Failed: {failed}, 💥 Errors: {errors}, ⚠️ Skipped: {skipped}")

if __name__ == "__main__":
    main()
