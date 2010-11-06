
from instant import inline_vtk
import vtk 

c_code1 = """
vtkObject* test1(char* filename){
  std::cout <<" Reading grid file  "<<std::endl; 
  vtkXMLUnstructuredGridReader* reader1 = vtkXMLUnstructuredGridReader::New();
  reader1->SetFileName(filename); 
  reader1->Update();
  vtkUnstructuredGrid* grid;
  grid= reader1->GetOutput();
  std::cout << "Number of cells "<<grid->GetNumberOfCells () <<std::endl; 
  return grid; 
}
"""

c_code2 = """
void test2(vtkUnstructuredGrid* grid){
  std::cout << "number of cells "<<grid->GetNumberOfCells () <<std::endl; 
  vtkCell* cell; 
  vtkPoints* points; 
  double x[3]; 
  for (int i=0; i<grid->GetNumberOfCells (); i++) { 
    cell = grid->GetCell (i);
    std::cout <<"cell no. "<<i <<" cell type "<< cell->GetCellType () <<std::endl; 
    points = cell->GetPoints ();
    std::cout <<"points->GetNumberOfPoints() "<< points->GetNumberOfPoints() <<std::endl;  
    for (int d=0; d<cell->GetNumberOfPoints(); d++){
	points->GetPoint(d, x); 
        std::cout <<" x " <<x[0]<<" "<<x[1]<<" "<<x[2]<<std::endl;  
    }
    vtkIdList* ids = cell->GetPointIds(); 
    for (int d=0; d<ids->GetNumberOfIds(); d++){
        std::cout <<" ids " <<ids->GetId(d) <<std::endl;  
    }
  }
}
"""


func1 = inline_vtk(c_code1, cache_dir="test_vtk")
func2 = inline_vtk(c_code2, cache_dir="test_vtk")

filename = "u2.vtu"
filename = "u000000.vtu"

mesh = func1(filename)
func2(mesh)


