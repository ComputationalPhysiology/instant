
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

A = vtkvmtk.vtkvmtkDoubleVector()
A.Allocate(12,1)
print A.ComputeNorm()
func(A)
print A.ComputeNorm()

