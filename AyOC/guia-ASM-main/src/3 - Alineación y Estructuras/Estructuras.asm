

;########### ESTOS SON LOS OFFSETS Y TAMAÑO DE LOS STRUCTS
; Completar las definiciones (serán revisadas por ABI enforcer):
NODO_OFFSET_NEXT EQU 8
NODO_OFFSET_CATEGORIA EQU 8
NODO_OFFSET_ARREGLO EQU 8
NODO_OFFSET_LONGITUD EQU 8
NODO_SIZE EQU 8
PACKED_NODO_OFFSET_NEXT EQU 8
PACKED_NODO_OFFSET_CATEGORIA EQU 8
PACKED_NODO_OFFSET_ARREGLO EQU 8
PACKED_NODO_OFFSET_LONGITUD EQU 8
PACKED_NODO_SIZE EQU 8
LISTA_OFFSET_HEAD EQU 8
LISTA_SIZE EQU 8
PACKED_LISTA_OFFSET_HEAD EQU 8
PACKED_LISTA_SIZE EQU 8

;########### SECCION DE DATOS
section .data

;########### SECCION DE TEXTO (PROGRAMA)
section .text

;########### LISTA DE FUNCIONES EXPORTADAS
global cantidad_total_de_elementos
global cantidad_total_de_elementos_packed

;########### DEFINICION DE FUNCIONES
;extern uint32_t cantidad_total_de_elementos(lista_t* lista);
;registros: lista[RDI]
cantidad_total_de_elementos:
	push rbp
	mov rbp, rsp

	; limpio contador (registro eax)
	xor eax, eax

	; cargo lista->head y la comparo con null
	mov rdi, [rdi]
	test rdi, rdi

	.loop:
	add eax, [rdi + NODO_OFFSET_LONGITUD]
	
	; comparo head->next con null, repito ciclo si no es null
	mov rdi, [rdi]
	test rdi, rdi
	jne .loop
	
	.epilogo:

	pop rbp
	ret

;extern uint32_t cantidad_total_de_elementos_packed(packed_lista_t* lista);
;registros: lista[?]
cantidad_total_de_elementos_packed:
	ret

