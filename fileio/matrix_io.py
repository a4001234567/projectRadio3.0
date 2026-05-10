import numpy as np
from typing import Iterable,Tuple,List,Mapping,Union,Dict
import zlib
import os
import re

'''
This module provides functions to read and write matrices in a specific format.
The format is as follows:
    The raw file is either a text file, with suffix .txt, or a compressed file, with suffix .zip
    The compressed file can be decompressed using zlib, where the decompressed content is identical to the text file.
    Lines in the file falls into three categories:
    1. Comments, which start with "//" and would be ignored by the reader program, can appear anywhere.
    2. Modifier, which start with ".", where:
        .sparse: indicates the file stores data in sparse format, where each data line contains coordinates, and sometimes values;
        .diff: indicats the file stores the location of non-zero elements, in the form of the next y-coordinate - current y-coordinate;
        .nonbinary (int:k): indicates the file stores matrix of non-binary field, the number k represents the field is F[k]
        multiple Modifier are allowed, latter modifier would override conflicting modifier. Modifier must come before the first data line.
    3. Data lines, which contain the actual data of the matrix. The data can be in three formats:
        1. Normal format, where each line contains a row of the matrix, separated by either commas, tabs, or spaces(define by the funtion _splitby);
        2. Sparse format, where each line contains a list of coordinates in the form of <x,y>, where x and y are the coordinates,
            when the file is nonbinary, the coorrdinates takes the form <x,y,v>, where x,y are the coordinates, and v are the values.
            Those structrue is identified using regular expression, so most deliminator is acceptable
        3. Diff format, each line contains a list of coordinates in the form of <x,y_diff>, where x is the row index, and y_diff is the difference
            between the current y-coordinate and the previous y-coordinate or 0.
            When non-binary field is used, the values are also included in the form of <x,y_diff,v>, where v is the value.
See the bottom for actual matrix example.
'''
__supportedMode = ('normal','sparse','diff')
class Writer:
    '''
    A class to write a matrix to a file.
    Example usage:
        with Writer(5,6,'test.txt',mode='normal',compress=False,comments=['test']) as writer:
            writer[0,0] = 1
            writer[1,1] = 1
            writer[4,5] = 1
            writer[0,2] = 1
            writer[0,4] = 1
    This will create a file named 'test.txt' with the following content:
    //test
    //H:5,W:6
    1 0 1 0 0 0
    0 1 0 0 0 0
    0 0 0 0 0 0
    0 0 0 0 0 1
    0 0 0 0 1 0
    '''
    def __init__(self,H:int,W:int,filename:str,mode:str='normal',fieldSize:int=2,comments:Union[List[str],None]=None,compress:bool=False):
        '''
        Initialize the Writer object.
            H: int, the height of the matrix
            W: int, the width of the matrix
            filename: str, the name of the file to write
            mode: str, the mode of the matrix, either 'normal', 'sparse', or 'diff'
            fieldSize:int the size of the finite field, default is 2
            comments: List[str], the comments to write to the file
            compress: bool, whether to compress the file
        '''
        self.filename:str = filename
        self.mode:str = mode
        self.compress:bool = compress
        self.comments:List[str] = comments if comments is not None else []
        self.fieldSize:int = fieldSize
        self._Xs:List[int] = []
        self._Ys:List[int] = []
        self.shape:Tuple[int,int] = H,W
        self._values:Dict[Tuple[int,int],int] = dict()
        if compress:
            assert self.filename.endswith('.zip'), f'Compressed file must have suffix .zip, got {self.filename}'
    
    def __enter__(self)->'Writer':
        return self
    
    def __setitem__(self,coord:Tuple[int,int],value:int):
        _coordX, _coordY = coord
        assert 0 <= _coordX < self.shape[0] and 0 <= _coordY < self.shape[1], f'Out of bound coordinate: {_coordX}, {_coordY} of {self.shape[0]}*{self.shape[1]} matrix'
        if 0 == value:
            return
        if 1 <= value:
            self._Xs.append(_coordX)
            self._Ys.append(_coordY)
            if 1 < value:
                self._values[(_coordX,_coordY)] = value
                assert value < self.fieldSize, f'Value {value} out of field F[{self.fieldSize}]'
    
    def __exit__(self,exc_type,exc_value,traceback):
        if exc_type is not None:
            return
        
        buffer = f'//H:{self.shape[0]},W:{self.shape[1]}\n'
        if self.fieldSize != 2:
            buffer += f'.nonbinary:{self.fieldSize}\n'
        for comment in self.comments:
            for line in comment.split('\n'):
                buffer += '//'+line+'\n'
        self.Xs, self.Ys = zip(*sorted(zip(self._Xs,self._Ys),key=lambda t:(t[0],t[1])))

        if self.mode == 'normal':
            temp_map:Mapping[int,set[int]] = dict() #Mapping of x to the list of non-zero entries in that row
            for x in range(self.shape[0]):
                temp_map[x] = set()
            for x,y in zip(self.Xs,self.Ys):
                temp_map[x].add(y)
            for x in range(self.shape[0]):
                # if x,y not in values, then it is 1 (1 is not recorded in values)
                buffer += ' '.join(map(lambda y:str(self._values.get((x,y),1) if y in temp_map[x] else 0),range(self.shape[1])))+'\n'
            
        elif self.mode == 'sparse':
            buffer += f'.sparse:{self.shape[0]}*{self.shape[1]}\n'
            prev_x = None
            for x,y in zip(self.Xs,self.Ys):
                if x != prev_x:
                    buffer += '\n'
                    prev_x = x
                if (x,y) not in self._values:
                    buffer += f'<{x},{y}>'
                else:
                    buffer += f'<{x},{y},{self._values[(x,y)]}>'
            buffer += '\n'
            
        elif self.mode == 'diff':
            buffer += f'.diff:{self.shape[0]}*{self.shape[1]}\n'
            prev_x = None
            prev_y = None
            for x,y in zip(self.Xs,self.Ys):
                if x != prev_x:
                    buffer += '\n'
                    prev_x = x
                    prev_y = None
                if prev_y is None:
                    y_diff = y
                else:
                    y_diff = y - prev_y
                    prev_y = y
                if (x,y) not in self._values:
                    buffer += f'<{x},{y_diff}>'
                else:
                    buffer += f'<{x},{y_diff},{self._values[(x,y)]}>'
            buffer += '\n'
        
        if self.compress:
            with open(self.filename,'wb') as file:
                file.write(zlib.compress(buffer.encode()))
        else:
            with open(self.filename,'w') as file:
                file.write(buffer)

def writer(filename:str, cont_matrix:np.ndarray, fieldSize:int=2, comments:Union[List[str],None]=None, mode:str='normal', compress:bool=False):
    '''
    Write a matrix to a file.
    The matrix must be a numpy array.
    The function will write the matrix to a file in the same format as the one written by the Writer class.
    '''
    H,W = cont_matrix.shape
    buffer = ''
    buffer += f'//H:{H},W:{W}\n'
    if fieldSize != 2:
        assert np.max(cont_matrix) < fieldSize, f'Value {np.max(cont_matrix)} out of field F[{fieldSize}]'
        buffer += f'.nonbinary:{fieldSize}\n'
    if comments is not None:
        for comment in comments:
             buffer += '//'+str(comment)+'\n'
    if mode == 'normal':
        for row in cont_matrix:
            buffer += (' '.join(map(lambda x:str(int(x)),row))+'\n')
    elif mode == 'sparse':
        buffer += f'.sparse:{H}*{W}\n'
        for x in range(H):
            for y in np.nonzero(cont_matrix[x])[0]:
                assert isinstance(x,(int,np.integer)),(x,type(x))
                assert isinstance(y,(int,np.integer)),(y,type(y))
                if cont_matrix[x][y] != 1:
                    buffer += f'<{x},{y},{int(cont_matrix[x][y])}>'
                else:
                    buffer += f'<{x},{y}>'
            buffer += '\n'
    elif mode == 'diff':
        buffer += f'.diff:{H}*{W}\n'
        for x in range(H):
            y_prev = None
            for y in np.nonzero(cont_matrix[x])[0]:
                assert isinstance(x,(int,np.integer)),(x,type(x))
                assert isinstance(y,(int,np.integer)),(y,type(y))
                if y_prev is None:
                    y_diff = y
                else:
                    y_diff = y - y_prev
                    y_prev = y
                if cont_matrix[x][y] != 1:
                    buffer += f'<{x},{y_diff},{int(cont_matrix[x][y])}>'
                else:
                    buffer += f'<{x},{y_diff}>'
            buffer += '\n'
    
    if compress:
        assert filename.endswith('.zip')
        with open(filename,'wb') as file:
            file.write(zlib.compress(buffer.encode()))
    else:
        with open(filename,'w') as file:
            file.write(buffer)

def _neglect(*string_set):
    def f(string:str):
        for i in string_set:
            string = string.replace(i,'')
        return string
    return f
_preprocessor = _neglect('[',']')

def _anysum(*to_add):
    if 0 == len(to_add):
        raise ValueError("Empty list to add")
    elif 1 == len(to_add):
        return to_add[0]
    return _anysum(to_add[0]+to_add[1],*to_add[2:])

def _splitby(string_set):
    def f(string:str):
        groups:List[str] = [string]
        for i in string_set:
            groups = _anysum(*(sub_string.split(i) for sub_string in groups))
        return groups
    return f
_splitter = _splitby((',','\t',' '))

pattern_finder = re.compile(r'<(\d+),(\d+)(?:,(\d+))?>')
sparse_header_finder = re.compile(r'.sparse:(\d+)\*(\d+)')
diff_header_finder = re.compile(r'.diff:(\d+)\*(\d+)')

def read_matrix(filename:str, returnFormat='MATRIX')->Union[np.ndarray,Tuple[np.ndarray,np.ndarray,np.ndarray,int,int]]:
    if not os.path.exists(filename):
        raise ValueError(f"{filename} does not exist")
    if filename.endswith('.zip'):
        with open(filename,'rb') as file:
            contents = zlib.decompress(file.read()).decode()
    else:
        with open(filename,'r') as file:
            contents = file.read()
    if returnFormat == 'MATRIX':
        return _read_matrix(contents)
    elif returnFormat == 'COORDS':
        return _readMatrixCoords(contents)
    else:
        raise ValueError(f"Unknown returnFormat: {returnFormat}")

def _readMatrixCoords(contents:str)-> Tuple[np.ndarray,np.ndarray,np.ndarray,int,int]:
    '''
    Read a matrix from a string and return the coordinates of non-zero entries.
    The string should be in the same format as the one written by the Writer class.
    The function will return a list of tuples, where each tuple is (x,y) coordinate of a non-zero entry.
    '''
    lines = iter(contents.split('\n'))
    mode = 'normal'
    fieldSize:int = 2
    xs:List[int] = []; ys:List[int] = []; values = []
    lineCnt = 0
    for line in lines:
        if not line: continue
        if line.startswith('//'): # Comment line
            pass
        elif line.startswith('.'): # Modifier line
            if line.startswith('.sparse'):
                mode = 'sparse'
                matchResult = sparse_header_finder.match(line)
                if matchResult is None:
                    raise ValueError(f"Invalid sparse header: {line}")
                h,w = matchResult.groups()
                h,w = map(int,(h,w))
            elif line.startswith('.diff'):
                mode = 'diff'
                matchResult = diff_header_finder.match(line)
                if matchResult is None:
                    raise ValueError(f"Invalid diff header: {line}")
                h_diff,w_diff = matchResult.groups()
                h_diff,w_diff = map(int,(h_diff,w_diff))
            elif line.startswith('.nonbinary'):
                fieldSize = int(line.split(':')[1])
            else:
                raise ValueError(f"Unknown modifier: {line.lstrip('.')}")
        else: # Data line
            if mode == 'normal':
                line = _splitter(_preprocessor(line.rstrip()))
                for y, val in ((_y,int(_val)) for _y,_val in enumerate(line) if _val != '0'):
                    xs.append(lineCnt)
                    ys.append(y)
                    assert val < fieldSize, f'Value {val} out of field F[{fieldSize}]'
                    values.append(val)
                lineCnt += 1
            elif mode == 'sparse':
                for matchResult in pattern_finder.findall(line):
                    if matchResult[-1]:
                        x,y,v = matchResult
                        x,y,v = map(int,(x,y,v))
                        xs.append(x)
                        ys.append(y)
                        assert v < fieldSize, f'Value {v} out of field F[{fieldSize}]'
                        values.append(v)
                    else:
                        x,y,_ = matchResult
                        x,y = map(int,(x,y))
                        xs.append(x)
                        ys.append(y)
                        values.append(1)
            elif mode == 'diff':
                prev_x = None; prev_y = None
                for matchResult in pattern_finder.findall(line):
                    if matchResult[-1]:
                        x,y_diff,v = matchResult
                        x,y_diff,v = map(int,(x,y_diff,v))
                    else:
                        x,y_diff,_ = matchResult
                        x,y_diff = map(int,(x,y_diff))
                        v = 1
                    if prev_x is None: prev_x = 0
                    if prev_y is None or prev_x != x:
                        prev_x = x
                        prev_y = 0
                    y = prev_y + y_diff
                    xs.append(x)
                    ys.append(y)
                    assert v < fieldSize, f'Value {v} out of field F[{fieldSize}] for position ({x},{y})'
                    values.append(v)
    if mode == 'sparse' or mode == 'diff':
        return np.array(xs),np.array(ys),np.array(values), h, w
    elif mode == 'normal':
        return np.array(xs),np.array(ys),np.array(values), max(xs)+1, max(ys)+1
    else:
        raise ValueError(f"Unknown mode: {mode}")

def _read_matrix(contents:str, outputFieldSize:bool=False)->Union[np.ndarray,Tuple[np.ndarray,int]]:
    '''
    Read a matrix from a string.
    The string should be in the same format as the one written by the Writer class.
    The function will return a numpy array of the matrix.
    '''
    lines = iter(contents.split('\n'))
    mode = 'normal'; board = []; fieldSize:int = 2
    for line in lines:
        if not line: continue
        if line.startswith('//'): # Comment line
            pass
        elif line.startswith('.'): # Modifier line
            if line.startswith('.sparse'):
                mode = 'sparse'
                matchResult = sparse_header_finder.match(line)
                if matchResult is None:
                    raise ValueError(f"Invalid sparse header: {line}")
                h,w = matchResult.groups()
                h,w = map(int,(h,w))
                board = np.zeros((h,w),dtype='int8')
            elif line.startswith('.diff'):
                mode = 'diff'
                matchResult = diff_header_finder.match(line)
                if matchResult is None:
                    raise ValueError(f"Invalid diff header: {line}")
                h_diff,w_diff = matchResult.groups()
                h_diff,w_diff = map(int,(h_diff,w_diff))
                board = np.zeros((h_diff,w_diff),dtype='int8')
            elif line.startswith('.nonbinary'):
                fieldSize = int(line.split(':')[1])
        else: # Data line
            if mode == 'normal':
                line = _splitter(_preprocessor(line.rstrip()))
                line = tuple(map(int,line))
                board.append(line)
            elif mode == 'sparse':
                assert isinstance(board,np.ndarray), f'board is not initialized, please check the file format'
                for matchResult in pattern_finder.findall(line):
                    if matchResult[-1]:
                        x,y,v = matchResult
                        x,y,v = map(int,(x,y,v))
                        assert v < fieldSize, f'Value {v} out of field F[{fieldSize}]'
                        board[int(x),int(y)] = v
                    else:
                        x,y,_ = matchResult
                        x,y = map(int,(x,y))
                        board[int(x),int(y)] = 1
            elif mode == 'diff':
                assert isinstance(board,np.ndarray), f'board is not initialized, please check the file format'
                prev_x = None; prev_y = None
                for matchResult in pattern_finder.findall(line):
                    if matchResult[-1]:
                        x,y_diff,v = matchResult
                        x,y_diff,v = map(int,(x,y_diff,v))
                    else:
                        x,y_diff,_ = matchResult
                        x,y_diff = map(int,(x,y_diff))
                        v = 1
                    if prev_x is None: prev_x = 0
                    if prev_y is None or prev_x != x:
                        prev_x = x
                        prev_y = 0
                    y = prev_y + y_diff
                    assert v < fieldSize, f'Value {v} out of field F[{fieldSize}] for position ({x},{y})'
                    board[x,y] = v
                    
    if isinstance(board,list):
        board = np.array(board,dtype='int8')
    if outputFieldSize:
        return board,fieldSize
    return board

'''
OUTPUT_FORMS = ('COORDS','ARRAY')
def from_coords_to_array(points:Iterable)->np.ndarray:
    board = None
    for x,y in points:
        if not board:
            w,h = points[0]
            board = np.zeros((h,w),dtype='int8')
        else:
            board[x][y] = 1
    return board
'''

def get_reader(output_form='ARRAY'):
    return read_matrix
    assert output_form in OUTPUT_FORMS
    if output_form == 'COORDS':
        return read_matrix
    elif output_form == 'ARRAY':
        def f(*args,**kargs):
            return from_coords_to_array(read_matrix(*args,**kargs))
        return f

def testSelf():
    assert _read_matrix(""".sparse:5*6
<0,0>,<1,1>,<4,5>
<0,2>,<0,4>
""").sum() == 5, "Test failed for sparse matrix reading"

def _testWriter():
    with Writer(5,6,'test.zip',mode='diff',fieldSize=3, compress=True,comments=['test']) as writer:
        writer[0,0] = 1
        writer[1,1] = 1
        writer[4,5] = 2
        writer[0,2] = 1
        writer[0,4] = 1
    print(read_matrix('test.zip'))

if __name__ == '__main__':
    _testWriter()
