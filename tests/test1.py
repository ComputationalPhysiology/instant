import Instant,os  

ext = Instant.Instant()

c_code = """
int hello(){
  printf("Hello World!\\n");
  return 2222;
}
"""

ext.create_extension(code=c_code,
                     module='test1_ext')

from test1_ext import hello

return_val = hello()
print return_val 


