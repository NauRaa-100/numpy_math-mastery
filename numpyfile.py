"""Tasks in Numpy library
"""
"""Part one Basics Fancy indexing
 ,indexing , boolean m slicing ..."""
#1 Group one
import numpy as np
my_array=np.array((10,20,30,40,50),ndmin=1 )
print(my_array)

#A
print(f'it has :{my_array.ndim} Dimensions') 
#B
print(f"it has {my_array.shape} shape") 
#c
print(f"Size: {my_array.size}")
#d
print(f"it has: {my_array.itemsize} Byte")
print('#'*10)
#------------------------------------
#2 
my_array=[[1, 2, 3],
 [4, 5, 6]]
#A
arr=np.array(my_array)
print(arr.dtype)

flo_my_array= np.array(my_array,dtype=float)
print(flo_my_array)
print('#'*30)
#-------------------------------------
# 3 Group two
#ِِA
my_array=np.zeros((3,3)) 
print(my_array)

#B
my_array=np.ones((4,2)) 
print(my_array)
print('#'*30)

#c
arr = np.array([99, 88, 77, 66])
reshaped = arr.reshape(2, 2)
print(reshaped)


#D

#مش عارفة بحجم 4 دى
my_array=np.array(('nora'))
print(my_array.itemsize)
#مصفوفة عشوائية 2×3 من القيم بين 0 و1

my_array=np.random.rand(3,2)
print(my_array)
print('#'*30)
#X مصفوفة أعداد صحيحة عشوائية 4×4 بين 10 و50
my_array=np.random.randint(10,51,size=(4,4))
print(my_array)
#---------------------------------
# Group three
#4

arr = np.array([[11, 12, 13],[21, 22, 23],[31, 32, 33]])
#A
print(arr[1,2])
#B
print(arr[0])
#c
print('#'*30)
print(arr[0, 1:3]) 

#d
print('#'*30)

print(arr[0:2])
print('#'*30)
#x

narr=arr.ravel()
for x in narr:
    if x >20:
        print(x, end=' ')
print('#'*30)
#------------------------------
#1 
my_array=[10,20,30,40]
arr=np.array(my_array)
"""Dimensions"""
print(arr.ndim)
""""Shape"""
print(arr.shape)
"""Data Type"""
print(arr.dtype)
"""Item Size"""
print(arr.itemsize)
#------------------------------
#2
my_array=[[5, 10], [15, 20], [25, 30]]
arr=np.array(my_array,ndmin=2)
"""number of elements"""
print(arr.size)
"""Item Size of elements"""
print(arr.itemsize)
"""Shape of matrix"""
print(arr.shape)
#-------------------------------
#3
"""Converting from float to integer numbers"""
data = [1.5, 2.5, 3.5]
arr=np.array(data,dtype=int)
print(arr.dtype)
#-------------------------------
#4
"""creating Zeros matrix , shape and dimensions"""
arr=np.zeros((4,3))
print(arr)
print(arr.shape)
print(arr.ndim)
#-------------------------------
#5
"""creating ones matrix and changing type to float"""
arr=np.ones((2,3),dtype=float)
print(arr)
#-------------------------------
#6
arr=np.array([9,9,9,9])
reshaped = arr.reshape(2, 2)
print(reshaped)

#-------------------------------
#7
arr=np.array([10,10,10,10,5,6,7,8,2,5,2,8,1,6,5,4])
arr.reshape(4,4)
print(arr)
#--------------------------------
#8
"""Generate random numbers"""
arr=np.random.randn(2,2) #randn مبتاخدش تابل زى الباقي كلو !!
print(arr)
#--------------------------------
#9
"""Generate random true numbers"""
arr=np.random.randint(0,100,size=(3,3))
print(arr)
print('#'*30)
#---------------------------------
#10
"""indexing & slicing"""
arr = np.array([[10, 20, 30],
                [40, 50, 60],
                [70, 80, 90]])


print(arr[1,2:3])
print(arr[0])
print(arr[0,1:2],arr[1,1:2],arr[2,1:2])
#--------------------------------------
#11
arr = np.array([[1, 5], [10, 20]])
print(arr[arr > 5])  


print('#'*30)

#--------------------------------------
#12
arr = np.array([5, 10, 15, 20, 25, 30])
print(arr[[1,3,5]])
#--------------------------------------
#1
my_list=[1,2,3,4,5,6]

my_array=np.array(my_list ,ndmin=2)
arr=my_array.reshape(3,2)

print(arr)
"""Dimension"""
print(arr.ndim)
"""Shape"""
print(arr.shape)
"""Size"""
print(arr.size)
"""Type"""
print(arr.dtype)
"""Item Size"""
print(arr.itemsize)
print('#'*30)

#------------------------------------
#2
"""Zeros Matrix"""
arr=np.zeros((2,3))
print(arr)

"""Ones Matrix"""
arr=np.ones((2,2),dtype=float)
print(arr)

"""only one value matrix"""

arr = np.array([9]*9).reshape(3,3)
print(arr)
print("#"*20)
#---------------------------------
#3
arr=np.random.randint(0,20,size=(4,4))
print(arr)
print(arr[1,3])
print(arr[:, 0]) 
print(arr[1,])
print('#'*30)

#----------------------------------
"""Boolean indexing """
print(arr[arr>10])
print(arr[arr % 2 ==0])
#----------------------------------
#5
print('#'*30)
arr = np.array([100, 200, 300, 400, 500, 600])
print(arr)
print(arr[[1,3,5]])
#--------------------------------------
#1

arr = np.arange(1, 10)
arr = arr.reshape(3, 3)
print(arr)

#----------------------------------------
#2
arr=np.array([7]*9,ndmin=3)

a=arr.reshape(3,3)
a.dtype='float32' 
print(a)
#----------------------------------------
#3
arr=np.random.randint(10,100,size=(5,4))
print(arr)
print('#'*10)
print(arr[2,])
print('#'*10)
print(arr[:,1])
print('#'*10)
print(arr[0,3])

#---------------------------------------
print(arr[arr % 2 ==0])
print(arr[arr > 50])
print('#'*10)

#---------------------------------------
#5
arr=np.array([10,20,30,40,50,60,70,80,90],ndmin=3)
a=arr.reshape(3,3)
print(a)
print('#'*10)
print(a[2,])
print('#'*10)
print(a[:,0])
print('#'*10)
print(a[a>=50])
print('#'*10)
#----------------------------------------
#6
arr= np.random.randint(0,9,size=(3,3))
print(arr)
print('#'*10)
print(arr[[0,2]])
print('#'*10)
#---------------------------------------
#7
arr= np.zeros(shape=(5,5))
print(arr)
print('#'*10)
print(arr[1:4])
print('#'*10)
#---------------------------------------
#8
arr=np.arange(1,13)
a=arr.reshape(3,4)
print(a)
print('#'*10)
a[a % 2 != 0] = 0
print(a)
#----------------------------------------
#1

arr= np.arange(10,100).reshape(9,10)
print(arr)
print('#'*20)
print(arr[3])
#-----------------------------------
#2
arr=np.random.randint(10,90,size=(4,4))
print(arr)
print('#'*20)
print(arr[:,0])
print('#'*20)
print(arr[3,])
print('#'*20)
print(arr[arr>50])
print('#'*20)
#---------------------------------
#3
arr=np.full((3,5),7)
print(arr)
print('#'*20)
#---------------------------------
#4
arr=np.random.randint(1,36,size=(6,6))
print(arr)
print('#'*20)
arr[arr %2!=0]=0
print(arr)
print('#'*20)
#----------------------------------
#5
arr=np.arange(5,100,step=5) 
print(arr)
print('#'*20)
print(arr[arr % 10==0])
print(arr.dtype)
print(arr.size)
print(arr.shape)
print(arr.ndim)
print('#'*20)

#------------------------------------
#6
arr=np.random.randint(1,12,size=(3,4))
print(arr)
print('#'*20)
print(arr[[0,2]])
print('#'*20)
#------------------------------------
#7
arr=np.full((3,3),9)
print(arr)
print('#'*20)
#-------------------------------------
#8
arr=np.arange(1,13)
print(arr)
print(arr.reshape(3,4))
arr[arr%2==0]=0
print(arr)
print('#'*20)
#------------------------------------
#9
arr=np.arange(1,26).reshape(5,5)
print(arr)
print('#'*20)
print(arr[2:4,1:3])
print('#'*20)
#------------------------------------
#10
arr=np.arange(1,13)
print(arr)
print('#'*20)
my_array=np.array(arr,ndmin=2)
print(my_array)
print(my_array.ndim)
my_array[my_array <= 6] = 0
my_array[my_array > 6] = 1
print(my_array)
print('#'*20)
#-------------------------------------
"""
Part Two Arthmetic Operations,
statistical Operations / statistical functions
, Broadcasting, Reshaping arrays,
concatination and spliting and Logical Operations /Boolean indexing

"""
#1
arr=np.arange(1,10 )
arr1=np.array(arr).reshape(3,3)
arr2=np.full((3,3),5)
print(arr1)
print('#'*20)
print(arr2)
print('#'*20)
print(arr1 + arr2)
print('#'*20)
print(arr1 - arr2)
print('#'*20)
print(arr1 * arr2)
print('#'*20)
print(arr1 / arr2)
print('#'*20)
print(arr1.__add__(arr2))
print('#'*20)
#-----------------------------------
#2
arr=np.random.randint(10,51,size=(4,4))
print(arr)
print(sum(arr)/arr.size)
print(arr.mean())
print(round(arr.std(),2))
print(arr.min())
print(arr.max())
print(arr.sum())
print(arr.dot(arr))
print('#'*20)
#-----------------------------------
#3
arr1=np.arange(10,22)
arr =np.array(arr1).reshape(3,4)
print(arr)
print(arr.argmin())
print(arr.argmax())
#-----------------------------------
#4
on_arr= [1,2,3,4,5]
arr1=np.arange(1,16)
arr=np.array(arr1).reshape(3,5)
print(sum(arr,10))
print(arr.dot(on_arr))
print('#'*20)
#-----------------------------------
#5
arr1=np.array([10,20,30,40]).reshape(1,4)
arr2=np.array([1,2,5,10],ndmin=1)
print(arr1)
print('#'*20)
print(arr2)
print('#'*20)
print(arr1 / arr2)
#------------------------------------
#6
arr=np.arange(1,13)
print(arr.reshape(3,4))
print(arr.reshape(2,6))
print(arr.flatten())
print('#'*20)
#-------------------------------------
#7
arr1=np.arange(1,7).reshape(2,3)
arr2=np.arange(7,13).reshape(2,3)
h=np.hstack((arr1 , arr2))
v= np.vstack((arr1, arr2))
print(arr1)
print(arr2)
print('#'*20)
#------------------------------------
#8
arr=np.random.randint(100,201 ,size=(2,6))
print(arr)
arr_s=np.hsplit(arr,3)
print(arr_s)
print('#'*20)
#---------------------------------
#9
arr=np.arange(1,17).reshape(4,4)
print(arr)
print('#'*20)
print(arr[arr>10])
print('#'*20)
#---------------------------------
#10
arr=np.arange(5,14).reshape(3,3)
mask=np.where(arr <9 , 0,1)
print(mask)
print('#'*20)
#---------------------------------
#11
arr=np.array([1,0,1,1,0,1]).reshape(6,)
print(np.any(arr))  # True
print(np.all(arr))  # False
#---------------------------------
#12
arr=np.arange(1,26).reshape(5,5)
arr[arr%2!=0]=-1
print(arr)
print("#"*30)
#--------------------------------
#1
arr1=np.arange(10,19).reshape(3,3)
arr2=np.full((3,3),5)
""" 1-Multiply"""
print(arr1.dot(arr2))
print("#"*30)
""" 2- Transposition """
tran=arr1.T
print(tran) 
print("#"*30)
print(tran + arr2)
print("#"*30)
print(tran - arr2)
print("#"*30)
#---------------------------------
#2
arr=np.random.randint(100,201,size=(4,4))
print(arr.mean())
tran=arr.T
print(tran)
print(tran.shape == arr.shape)
print(arr-tran)
print(arr/arr.std())
print("#"*30)
#---------------------------------
#3
arr=np.random.randint(50,101,size=(4,6))
h=np.hsplit(arr,3)
v=np.vsplit(arr,2)
print(arr)
print("#"*30)
print(v)
print("#"*30)
print(h)
print("#"*30)
print(np.array(h).T)
print(np.array(v).T)

#----------------------------------
#4
arr=np.arange(1,25).reshape(4,6)
print(arr.T)
print('#'*30)
chng=np.array_split(arr,3,axis=0)
print(chng)
merged = np.vstack(chng[:2])  
print(merged)

print('#'*30)
#----------------------------------
#5
arr=np.array([4, 8, 0, 2, 0, 7, 0, 5])
isol=arr[arr !=0]
print(isol)
print('#'*30)
#-----------------------------------
#1
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([10, 20, 30])
"""Addition!"""
print(arr1 + np.array([[10,20,30]])) 
print(arr1.__add__(arr2))
print('#'*30)
"""Multiply"""
print(arr1 *2)
print('#'*30)
"""Statistics"""
print(arr1.mean())
print(np.median(arr1))
print(round(arr.std(),2))
print(arr1.sum()/arr1.size)
print('#'*30)
#------------------------------------
#2
arr=np.arange(12).reshape(3,4)
"""Transposition"""
tran=arr.T
print(tran)
print('#'*30)
tran_arr=np.array(tran,ndmin=1).reshape(12,)
print(tran_arr)
odd=tran[tran%2 !=0]
print(odd.size)
print('#'*30)
#------------------------------
#3
"""slicing and spilt"""
arr=np.arange(1,25).reshape(4,6)
spl_arr=np.array_split(arr,3,axis=0)
print(spl_arr)
print('#'*30)
part_spl=spl_arr[0:1]
print(part_spl)
new_part = np.array([[1, 2, 3, 4, 5, 6]])
stacked = np.vstack([arr, new_part])
print('#'*30)
#--------------------------------
#4
"""Filtering & conditions"""
arr=np.array([[8,12,5],[3,18,7]])
print(arr[arr>6])
print(np.where(arr < 6, 0, arr))
print(np.all(arr[arr>2]))
print(np.any(arr[arr<5]))
#-------------------------------
#5
"""Random Stats"""
arr=np.random.randint(50,100,size=(5,4))
print(arr.max())
print(arr.min())
print(arr.mean())
print(arr.sum()/arr.size)
print(arr.std())
print(arr.argmax())
print(arr.argmin())
#--------------------------------
#1
arr=np.arange(1,21)
even=arr[arr%2==0]
print(f"even numbers: {even}")
print('#'*30)
print(f"sum: {arr.sum()}")
print(f"Median: {np.median(arr)}")
print('#'*30)
#-------------------------------
#2
arr=np.random.randint(10,101,size=(4,5))
print(f"Highest number is:{arr.max()}")
print(f"Lowest number is :{arr.min()}")
print('#'*30)
#-------------------------------
#3
arr=np.arange(1,37).reshape(6,6)
tran=arr.T
print(tran)
print(f"Even Numbers after transposition: {arr[arr % 2 ==0]}")
print('#'*30)
#--------------------------------
#4
arr1=np.arange(1,10).reshape(3,3)
arr2=np.full((3,3),5)
print(arr1 + arr2)
print('#'*30)
print(arr1.dot(arr2))
print('#'*30)
#--------------------------------
#5
arr=np.arange(12)
print(arr.reshape(3,4))
print('#'*30)
tran=arr.T
print(tran)
print('#'*30)
print(tran.reshape(-1))
print('#'*30)
#---------------------------------
#6
arr=np.random.randint(0,51,size=(4,5))
cond=arr[arr>25]
print(cond)
print('#'*30)
print(cond.sum())
print(cond.mean())
#----------------------------------
#7
arr=np.arange(24).reshape(4,6)
spl_arr=np.split(arr,2,axis=0)
print(spl_arr)
print('#'*30)
print(np.vstack(spl_arr))
#الدمج مش عارفاة ولا عارفة شروطة ومش شغال اشرحهولى 
#----------------------------------
#8
arr=np.arange(9).reshape(3,3)
print(arr.argmax())
print(arr.argmin())
print(arr.max())
print(arr.min())
print(arr.sum())
print(arr.std())
print(arr.mean())
print(arr.sum()/arr.size)
#----------------------------------
#9
arr=np.array([4, 6, 8, 10, 12])
cond1=arr[arr<7]
print(cond1)
print('#'*30)
print(cond1.astype(int)) 
#----------------------------------
#10
arr=np.arange(0,24).reshape(4,6)
print(arr)
print('#'*30)
print(arr[1:3,1:5])
print('#'*30)
print(np.where(arr>15,999,arr))
print('#'*30)
#----------------------------------
#1
arr=np.random.randint(0,101,size=(4,5))
even=arr[arr%2==0]
print(even)
print(even.mean())
#----------------------------------
#2
arr=np.arange(1,37).reshape(6,6)
tran=arr.T
print(tran)
print(tran.sum())
print('#'*30)
print(tran[tran>20])
print('#'*30)
#----------------------------------
#3
arr1=np.arange(1,10).reshape(3,3)
arr2=np.full((3,3),7)
sum_arr=arr1 + arr2
print(sum_arr)
print('#'*30)
print(arr1.dot(arr2))
print('#'*30)
median_arr=sum_arr.mean()
print(sum_arr[sum_arr>median_arr])
print('#'*30)
#-----------------------------------
#4
arr=np.arange(32).reshape(4,8)
spl_arr=np.split(arr,4,axis=0)
print(spl_arr)
print('#'*30)
spl_arr_h=np.split(arr,4,axis=1)
v=np.vstack(spl_arr[0:2])
print(v)
print('#'*30)
h=np.hstack(spl_arr_h[1:3])
print(h)
#-----------------------------------
#5
arr=np.random.randint(10,100,size=(5,5))
print(round(arr.std(),2))
median_arr=np.median(arr)
print(median_arr)
abs_arr=np.abs(arr)
print(abs_arr - median_arr)
#-----------------------------------
#6
arr=np.arange(1,19).reshape(3,6)
print(arr.sum())
median_arr=arr.mean()
mask=np.where(arr>median_arr,'pass','fail')
print(mask)
print('#'*30)
#------------------------------------
#7
arr= np.arange(1,31)
reshaped=arr.reshape(5,6)
print(reshaped[2:4,1:5])
reshaped[0:,0:]=0
print(reshaped)
#-----------------------------------
#8
arr=np.array([[-1, 4, 7], [0, -2, 6], [8, -3, 1]])
bol=arr.astype(bool)
print(bol)
print(arr * bol)
print('#'*30)
#-----------------------------------
#9
arr=np.random.randint(1,37,size=(6,6))
median_arr=np.median(arr)
print(arr - median_arr)
print('#'*20)
print(arr[arr==median_arr])
print('#'*20)
#------------------------------------
#10
arr=np.arange(1,13).reshape(3,4)
med_large=np.median(arr)
new_arr=arr[arr>med_large]
print(new_arr.size)
col_means = arr.mean(axis=0)
cols_mask = col_means > med_large
filtered_cols = arr[:, cols_mask]
print(filtered_cols)
#-----------------------------------
#1
a = np.array([[1, 2, 3]])
b = np.array([[4, 5]])

b_e=np.pad(b,((0,0),(0,1)),mode='constant',constant_values=0)
print(np.vstack((a , b_e)))
print('#'*20)
#---------------------------------
#2
x = np.array([[10], [20], [30]])
y = np.array([[1, 2, 3]])
y_p=np.pad(y,((0,2),(0,0)),mode='constant',constant_values=0)
print(np.hstack((x,y_p))) 
print('#'*30)
#--------------------------------------
#3
m = np.array([[5, 6]])
n = np.array([[7, 8, 9]])

m_p=np.pad(m,((0,0),(0,1)),mode='constant',constant_values=0)
merged=np.vstack((m_p,n))
print(merged)
print(merged[0].sum())
print(merged[1].sum())
print('#'*30)
#----------------------------------
#4
a = np.array([1, 2])
b = np.array([[3], [4]])
print(a.shape)
print(b.shape)
print(a.ndim)
print(b.ndim)
reshaped_a=np.array(a,ndmin=2)
a_p=np.pad(reshaped_a,((0,1),(0,0)),mode='constant',constant_values=0)
print(np.hstack((a_p , b)))
print('#'*30)
#------------------------------------
#5
arr1 = np.array([[1, 2]])
arr2 = np.array([[3], [4]])

print(arr1.ndim)
print(arr2.ndim)
print(arr1.shape)
print(arr2.shape)

arr1_p=np.pad(arr1,((0,1),(0,0)),mode='constant',constant_values=0)
print(np.hstack((arr1_p,arr2)))
print('#'*30)
#----------------------------------
"""

Part Three:
linspace ,
nan array,
view() and copy()
inversing array,
np.linalg.det(),
eigenvalues & eigenvectors 
np.linalg.eig()

"""

#1
arr=np.linspace(5,20,7)
print(arr)
print('#'*30)
#----------------------------------
#2
arr = np.array([2, np.nan, 5, np.nan, 9])
print(np.isnan(arr))
#-----------------------------------
#3
print(round(np.nanmean(arr),2))
print(np.nansum(arr))
#-----------------------------------
#4
arr = np.array([10, 20, 30])
arr_view = arr.view()
arr_copy = arr.copy()

arr[0] = 100
print(arr_view)
print(arr_copy)

"""

## arr.copy() creates an independent copy that does NOT reflect changes from the original array.
## arr.view() is a shallow copy, so it reflects changes in the original array.

"""
print('#'*30)
#----------------------------------
#5
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

print(a.dot(b))
print('#'*30)
#----------------------------------
#6
print(a@b)
print('#'*30)
#---------------------------------
#7
a = np.array([[4, 7],[2, 6]])
print(np.linalg.inv(a))
print('#'*30)
#--------------------------------
#8
print(round(np.linalg.det(a),2))
print('#'*30)
#--------------------------------
#9
a = np.array([[2, 1], [1, 2]])
values_vector_arr=np.linalg.eig(a)
print(values_vector_arr)
print('#'*30)
#---------------------------------
#10
arr=np.linspace(1,9,9).reshape(3,3)
print(np.linalg.det(arr))
print('#'*30)
print(np.linalg.eig(arr))
print('#'*30)
#----------------------------------
#1
arr=np.linspace(10,100,10)
print(arr)
print('#'*30)
reshaped=arr.reshape(2,5)
print(reshaped)
print('#'*30)
reshaped_p=np.pad(reshaped,((0,3),(0,0)),mode='constant',constant_values=0)
print(reshaped_p)
print(reshaped_p.ndim)

#-----------------------------------
#2
arr = np.array([[np.nan, 2, 3], [4, np.nan, 6], [7, 8, np.nan]])
mean_arr=np.nanmean(arr)
print(mean_arr)
print(np.where(np.isnan(arr)==True , mean_arr,arr ))
print('#'*30)
#-----------------------------------
#3
arr1 =np.random.randint(1,11,size=(3,3))
arr2 =np.random.randint(1,11,size=(3,3))
result_arr=arr1@arr2
print(result_arr)
print('#'*30)
print(arr1.dot(arr2))
print('#'*30)
print(np.linalg.inv(result_arr))
print('#'*30)
print(np.linalg.det(result_arr))
print('#'*30)
print(np.linalg.eig(result_arr).eigenvalues)  
print('#'*30)
#--------------------------------------
#4
arr=np.arange(5)
arr_view=arr.view()
arr_copy=arr.copy()
arr[0]=1000
print(arr_view) #[1000    1    2    3    4] 
print('#'*30)
print(arr_copy) #[0 1 2 3 4]
print('#'*30)
#--------------------------------------
#5
arr=np.arange(6)
arr.dtype='float'
print(arr)
print('#'*30)
arr[1]=np.nan
arr[2]=np.nan
print(np.nansum(arr))
print(np.nanmean(arr))
filtered_arr = arr[~np.isnan(arr)]
print(filtered_arr)
#--------------------------------------
#6
arr=np.array([
[4, 2, 0],
[2, 4, 2],
[0, 2, 4]
])
print(np.linalg.eig(arr))
print(np.linalg.det(arr))
print(np.linalg.eig(arr).eigenvectors)
print('#'*20)
print('#'*30)
#------------------------------------
#7
P = np.array([
    [0.5, 0.3, 0.2],
    [0.2, 0.5, 0.3],
    [0.3, 0.2, 0.5]
])
print(np.linalg.eig(P).eigenvalues)
print('#'*20)
print(np.max(np.linalg.eig(P).eigenvalues))
print('#'*20)
print(np.min(np.linalg.eig(P).eigenvalues))
print('#'*30)

#--------------------------------------
"""Exam 'General Numpy'"""
#1
arr=np.arange(10,100)
#element length
print(f'Length of elements: {arr.size}')
#reshaping array
print(f'Reshaped array: {arr.reshape(5,18)}')
#filter any falsy value
filtered_arr=arr[~np.isnan(arr)]
print(filtered_arr)
print('----------Seperate-----------')
#--------------------------------
#2
arr=np.arange(12)
#Converted to 2 Dimensions
my_array=np.array(arr,ndmin=2)
print(f'2D: {my_array}')
print('----------Seperate-----------')
#reshaping array to 3x4
my_array=np.array(arr,ndmin=2).reshape(3,4)
print(f'Reshaped array: {my_array}')
print('----------Seperate-----------')
#flating array to 1Dimension
print(f'1D :{np.ravel(my_array)}')
print('----------Seperate-----------')
#--------------------------------
#3
arr=np.arange(1,11)
arr_copy=arr.copy()
arr_view=arr.view()
arr[5]=1000
print(arr) #directly changed by update indexing
print('----------Seperate-----------')
print(arr_copy) #its a shallow copy nothing change into original array
print('----------Seperate-----------')
print(arr_view) # it can change what updating in original array
print('----------Seperate-----------')

#--------------------------------
#4
arr=np.array([1 ,np.nan , 2, 3,np.nan, 4, 5 ])
#mean of array
mean_arr=np.nanmean(arr)
print(mean_arr)
print('----------Seperate-----------')
print(np.where(np.isnan(arr),mean_arr,arr))
print('----------Seperate-----------')
#--------------------------------
#5
arr=np.random.randint(0,50,size=(5,5))
#convert to 2 Dimensions
my_array=np.array(arr,ndmin=2)
print('----------Seperate-----------')
#max values
print(f'Max value: {my_array.max()}')
print('----------Seperate-----------')
#Odd values
print(f'Odd values: {my_array[my_array %2 !=0]}')
print('----------Seperate-----------')
#--------------------------------
#6
arr=np.random.randint(-5,31,size=(5,15))
#Values less than 10 into zero 
print(np.where(arr <= 10, 0, 1))
print('----------Seperate-----------')
#---------------------------------
#7
arr1=np.arange(10)
arr2=np.arange(1,11)
#H merging
H=np.hstack((arr1,arr2))
print(H)
print('----------Seperate-----------')
#V merging
V=np.vstack((arr1,arr2))
print(V)
print('----------Seperate-----------')
#spilt H
arr_sp_h=np.array_split(H,3,axis=0)
print(arr_sp_h)
print('----------Seperate-----------')
#spilt V
arr_sp_v=np.array_split(V,3,axis=1)
print(arr_sp_v)
print('----------Seperate-----------')
#-------------------------------
#8
arr=np.random.randint(1,17,size=(4,4))
#Mean for every raw
print(np.mean(arr,axis=0))
#std for each col
print((np.std(arr,axis=1)))
#maxiumam value
print(arr.max())
#lowest index 
print(arr.argmin())
print('----------Seperate-----------')
#---------------------------------
#9
arr=np.arange(1,5).reshape(2,2)
#deter of array
print(np.linalg.det(arr))
print('----------Seperate-----------')
#invers array
try:
    inve=np.linalg.inv(arr)
    print(inve)
except:
    print("Can't be inversed")

print('----------Seperate-----------')
#---------------------------------
#10
arr1=np.arange(1,10).reshape(3,3)
arr2=np.arange(0,9).reshape(3,3)
#suring that 2 arrays have same size
print(arr1.size==arr2.size)
#multiply arrays
print(f'Way one: {arr1@arr2}')
print('----------Seperate-----------')
print(f'Way Two: {arr1.dot(arr2)}')
#---------------------------------
#11
arr=np.array([1,2,3,4],ndmin=2).reshape(2,2)
result=np.linalg.eig(arr)
#eigvalues
print(f'Eigvalues: {result[0]}')
print('----------Seperate-----------')
#eigvectors
print(f'EigVectors: {result[1]}')
print('----------Seperate-----------')
#----------------------------------
#12
arr=np.arange(1,10).reshape(3,3)
#padding full raw
arr_update=np.pad(arr,((0,1),(0,0)),mode='constant',constant_values=0)
#updating new values
arr_update[3,0]=10
arr_update[3,1]=20
arr_update[3,2]=30
print(arr_update)
print('----------Seperate-----------')
#----------------------------------
#13
arr=np.random.randint(0,100,size=(4,4))
#only values can divide 10
print(arr[arr%10==0])
print('----------Seperate-----------')
#----------------------------------
#14
arr=np.arange(1,10).reshape(3,3)
new_arr=np.array(arr,ndmin=1).ravel()
inds=[0,4,8]
np.take(new_arr,inds)
np.put(new_arr,inds,[99])
print(new_arr)
print('----------Seperate-----------')
#-----------------------------------
#15
arr=np.arange(10,100).reshape(10,9)
my_array=np.array(arr,ndmin=2)
#mean of students
mean_arr=np.mean(my_array,axis=0)
# clever student
print(mean_arr.max())
print('----------Seperate-----------')























































































































































































































































































































































































































