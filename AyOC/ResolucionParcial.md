Lo primero que pienso es que malloco va a trabajar sobre una sola tabla de página
A medida que se le pide memoria se crean nuevas entradas
La memoria fisica sale del area libre de tareas, 0x00400000
La memoria virtual a partir de 0xA10C0000

int 100 malloco codigo privilegiado, llamado por usuarios
int 101 chau    codigo privilegiado, llamado por usuarios

```c
IDTENTRY3(100);
IDTENTRY3(101);
```

Definimos que el parámetro se pasará a través del registro EAX en ambas syscalls.
Luego, el código en C lo lee desde la pila
```x86asm
global _isr100:
    pushad

    push EAX
    call malloco
    pop EAX

    popad
    iret
```
```x86asm
global _isr101:
    pushad

    push EAX
    call chau
    pop EAX

    popad
    iret
```
Ahora bien, si malloco está siendo usado por primera vez en esa tarea, le damos una tabla nueva con una sola entrada. Esto siempre le da 4kb.
Luego va asignando a medida que lo vaya accediendo en memoria, no cuando lo necesite. Cuando la tabla esté llena y no pueda asignar mas, devuelve NULL.

malloco necesita el cr3 de la tarea y mmu_next_user_page() para mapear nuevas páginas. El parámetro de dirección virtual dependerá de los pedidos previos de memoria (o del parámetro que nos pasan?) y los atributos siempre serán (MMU_U | MMU_W | MMU_P)

El page_fault_handler debe tener una condición para comprobar si la memoria virtual accedida se comprende entre 0xA10C0000 y (0xA10C0000+PAGE_SIZE*1024).
Si es cierto, verifico la estructura del kernel y mapeo la página de forma contigua.

Si una tarea pide 5 bytes de memoria, malloco le da un puntero al principio de pagina. Si vuelve a pedir memoria, en realidad podría darle el puntero que le di anteriormente y sumarle 5? O tengo que darle el puntero a la página siguiente y se pierde la diferencia pedida anteriormente?

Asumo primero el segundo caso luego de hablarlo con los ayudantes, si me da tiempo implemento el primer caso que es mas eficiente y no fragmento la memoria.

EDIT: luego de que aclararan los ayudantes, asumo que size es congruente a 0 mod 4096.

Si el acceso es invalido (fuera de su tabla de páginas) libero toda su memoria reservada y hago jmp far a la siguiente tarea.

Puedo usar una matriz con filas representando el índice de la tarea correspondiente al array de sched_tasks[MAX_TASKS] y columnas que representen
las páginas asignadas (1024 en total). El valor default de las páginas es 0 y llevará el total de bytes solicitados en esa página.

ESTO DE ARRIBA NO HACE FALTA!!!

Con un solo array de tamaño [MAX_TASKS] se puede trackear de la misma forma. Llevo el total de bytes solicitados por tarea y solo cuando supera un multiplo
de 1024 asigno una nueva página(si es que fue accedida). Lo llamo uint16_t malloco_bserver[MAX_TASKS] = {0}; y tendrá alcance global para todo el kernel.
Leo la variable global current_task del scheduler para ver que tarea estoy ejecutando y así modificarlo.

Pero no es info suficiente para chau...
Array de arrays de malloco_tracker_t de alcance global. Filas representan indice de tarea correspondiente al array de sched_tasks[MAX_TASKS] y columnas los pedidos de malloco.
hago un define para aclarar #define VALOR_GRANDE_SUFICIENTE_PARA_TODO_EL_SISTEMA ??? asumo que me lo dan y los miembros del struct pedidos ni marcados no pueden exceder ese tamaño.
```c
typedef struct {
  uint32_t usos = 0;                // "tamaño" actual del array pedidos
  uint32_t* pedidos[];              // Cantidad de bytes * 4096 pedidas en cada iteracion
  uint32_t total = 0;               // Suma total de memoria solicitada hasta el momento
  uint32_t* marcados[] = {0};       // bitmap correspondiente a pedidos[]. 1 si se marcó para borrar, 0 sino
} malloco_tracker_t;

malloco_tracker_t ultra_super_array[MAX_TASKS][]= {0};
```

Creamos el #define MALLOCO_START_VIRT 0xA10C0000
```c
void* malloco(size_t size){
    if((ultra_super_array[current_task].total + size) < PAGE_SIZE*1024){ // Registro el pedido si en total se pidieron menos de 4mb
        ultra_super_array[current_task].pedidos[ultra_super_array[current_task].usos] = size;
        ultra_super_array[current_task].usos++;
        ultra_super_array[current_task].total+=size;

        return (MALLOCO_START_VIRT + ultra_super_array[current_task].total - size);
    }
    return NULL;    // De lo contrario digo que no puedo
}
```
malloco por si solo no hace mucho, solo lleva la cuenta de cuanto se le pidió y que pedidos se realizaron.

Ahora modifico el page handler del tp.
```c
bool page_fault_handler(vaddr_t virt) {
  uint32_t cr3 = rcr3();
  bool res = false;
  // Si está en el bloque de memoria virtual asignada para hacer malloco, asigno memoria
  if(virt >= MALLOCO_START_VIRT && virt <= (MALLOCO_START_VIRT+PAGE_SIZE*1024)){ 
        mmu_map_page(cr3,virt,mmu_next_free_user_page(),(MMU_U | MMU_W | MMU_P));
    res = true;
  }
  return res;
}
```
Modifico también _isr14 y añado que cuando page_fault_handler devuelve FALSO realizo lo siguiente
```x86asm
mov EAX,current_task
call chau_all
push EAX
call sched_disable_task ; lo tengo que hacer en este orden porque sino no puedo acceder a la tarea
pop EAX
call sched_next_task 
cmp ax, 0
je .fin

str bx
cmp ax, bx
je .fin

mov word [sched_task_selector], ax
jmp far [sched_task_offset]
```

donde chau_all es...
```c
void chau_all(void){
    uint32_t ultima_direccion_virtual = MALLOCO_START_VIRT;
    for(uint32_t i=0; i<ultra_super_array[current_task].usos;i++){
        ultra_super_array[current_task].marcados[i] = 1;
    }
}
```


```c
void chau(void* ptr){
    uint32_t offset = &ptr - MALLOCO_START_VIRT;
    uint32_t contador = 0;
    uint32_t i=0;
    // Tengo que ver cuanto se pidio al momento de haber llamado a malloco
    while( contador != offset){
        contador += ultra_super_array[current_task].pedidos[i];
        i++;
    }
    contador /= PAGE_SIZE;
    ultra_super_array[current_task].marcados[i] = 1;
    // Ahora contador tiene la cantidad de paginas a marcar para desmapear mas adelante;

}
```
```c
typedef struct {
  uint32_t usos = 0;
  uint32_t* pedidos[];
  uint32_t total = 0;
} malloco_tracker_t;

```

Tarea barrendero codigo privilegiado, llamado por el kernel cada 100 ticks
la añado a la gdt con el constructor de tss usado en el tp y será ubicada en gdt[20] =

```c
gdt_entry_t tss_gdt_entry_for_task(tss_t* tss) {
  return (gdt_entry_t) {
    .g = 0,
    .limit_15_0 = sizeof(tss_t) - 1,
    .limit_19_16 = 0x0,
    .base_15_0 = GDT_BASE_LOW(tss),
    .base_23_16 = GDT_BASE_MID(tss),
    .base_31_24 = GDT_BASE_HIGH(tss),
    .p = 1,
    .type = DESC_TYPE_32BIT_TSS, // 0x9
    .s = 0,
    .dpl = 0,
    .avl = 0,
    .l = 0,
    .db = 0
  };
}
```
```c
tss_t tss_create_kernel_task(paddr_t code_start) {
    next_
  return (tss_t) {
    .cr3 = mmu_init_task_dir(code_start),
    .esp = TASK_STACK_BASE,
    .ebp = TASK_STACK_BASE,
    .eip = TASK_CODE_VIRTUAL,
    .cs = GDT_CODE_0_SEL,
    .ds = GDT_DATA_0_SEL,
    .es = GDT_DATA_0_SEL,
    .fs = GDT_DATA_0_SEL,
    .gs = GDT_DATA_0_SEL,
    .ss = GDT_DATA_0_SEL,
    .ss0 = GDT_DATA_0_SEL,
    .esp0 = mmu_next_free_kernel_page()+PAGE_SIZE, 
    .eflags = EFLAGS_IF,                           
  };
}
```
donde TASK_STACK_BASE y TASK_CODE_VIRTUAL son regiones de memoria que no entran en conflicto con otras ya definidas previamente.
Supongo que puedo ubicar code_start en algún lado del area libre de kernel, asumo que barrendero pesa menos de 4kb y con una pagina basta.
code_start = mmu_next_free_kernel_page();

Modificamos _isr32, nuestra interrupcion de clock en el tp y le agregamos lo siguiente.

- Una variable global inicializada en 0
- Le sumamos uno a la misma en cada tick
- Añadimos un compare antes de cambiar de tarea
- Cuando llega a 100 llamamos a nuestra tarea de kernel para barrer la memoria marcada

```x86asm
counter_sweep: dw 0

_isr32:
    inc WORD[counter_sweep]
    cmp WORD[counter_sweep],100
    je BARRENDERO_SEL:0
```
donde BARRENDERO_SEL = (20 << 3)

y barrendero ejecuta:

```c
void barrendero(void){
    // Recorro todas las tareas con sched_tasks[i], obtengo su cr3
    // Por cada tarea accedo a ultra_super_array[i], recorro pedidos[] hasta usos-1
    // Desmapeo las direcciones de memoria cuando el mismo indice en marcados[] es 1
    // Seteo el mismo indice en marcados[] en 0
    uint32_t cr3;
    uint32_t cantidad_paginas;

    for(uint32_t i=0;i < MAX_TASKS;i++){
        cr3 = obtener_CR3_tarea(i);
        uint32_t cursor_direccion = MALLOCO_START_VIRT;
        for(uint32_t j=0; j <ultra_super_array[i].usos;j++){
            cursor_direccion += ultra_super_array[i].pedidos[j]; // Sumo los bloques pedidos como offset, estén marcados o no
            if(ultra_super_array[i].marcados[j] = 1){
                cantidad_paginas = ultra_super_array[i].pedidos[j] / PAGE_SIZE;
                //Si, de verdad estoy haciendo un triple for anidado...
                for(uint32_t k=0; k < cantidad_paginas;k++){
                    mmu_unmap_page(cr3,(vaddr_t*)cursor_direccion);
                }
            }
        }
    }
}
```

donde obtener_CR3_tarea() es:

```c
uint32_t obtener_CR3_tarea(uint32_t indice){
    //Accedo a la tss de la tarea, leo el cr3
    //Todo únicamente sabiendo el índice de la tarea respecto a sched_tasks
    uint16_t gdt_indice = sched_tasks[indice].selector >> 3;
    gdt_entry_t* tss_descriptor = &gdt[gdt_indice];

    uint32_t dir_base = (tss_descriptor.base_31_24 << 24) | (tss_descriptor.base_23_16 << 16) | (tss_descriptor.base_15_0);

    tss_t* tss = (tss_t*) dir_base;
    return tss.cr3;
}
```

Eso es todo amigos. Si me olvidé de algo voy a estar muy triste