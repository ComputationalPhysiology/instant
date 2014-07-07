from __future__ import print_function

from instant import inline_vmtk 
import vmtk
from vmtk import vtkvmtk

code = """
void test(vtkvmtkDoubleVector* a) {
  for (int i=0; i<a->GetNumberOfElements(); i++) {
    a->SetElement(i,i); 
  }
}
"""



func = inline_vmtk(code, cache_dir="test_vmtk")

v = vtkvmtk.vtkvmtkDoubleVector()
v.Allocate(12, 1)
print("norm of v ", v.ComputeNorm())
func(v)
print("norm of v after test ", v.ComputeNorm())

