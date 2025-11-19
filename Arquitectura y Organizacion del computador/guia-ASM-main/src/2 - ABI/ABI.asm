extern sumar_c
extern restar_c
;########### SECCION DE DATOS
section .data

;########### SECCION DE TEXTO (PROGRAMA)
section .text

;########### LISTA DE FUNCIONES EXPORTADAS

global alternate_sum_4
global alternate_sum_4_using_c
global alternate_sum_4_using_c_alternative
global alternate_sum_8
global product_2_f
global product_9_f

;########### DEFINICION DE FUNCIONES
; uint32_t alternate_sum_4(uint32_t x1, uint32_t x2, uint32_t x3, uint32_t x4);
; parametros: 
; x1 --> EDI
; x2 --> ESI
; x3 --> EDX
; x4 --> ECX
alternate_sum_4:
  sub EDI, ESI
  add EDI, EDX
  sub EDI, ECX

  mov EAX, EDI
  ret

; uint32_t alternate_sum_4_using_c(uint32_t x1, uint32_t x2, uint32_t x3, uint32_t x4);
; parametros: 
; x1 --> EDI
; x2 --> ESI
; x3 --> EDX
; x4 --> ECX
alternate_sum_4_using_c:
  ;prologo
  push RBP ;pila alineada
  mov RBP, RSP ;strack frame armado
  push R12
  push R13	; preservo no volatiles, al ser 2 la pila queda alineada

  mov R12D, EDX ; guardo los parámetros x3 y x4 ya que están en registros volátiles
  mov R13D, ECX ; y tienen que sobrevivir al llamado a función

  call restar_c 
  ;recibe los parámetros por EDI y ESI, de acuerdo a la convención, y resulta que ya tenemos los valores en esos registros
  
  mov EDI, EAX ;tomamos el resultado del llamado anterior y lo pasamos como primer parámetro
  mov ESI, R12D
  call sumar_c

  mov EDI, EAX
  mov ESI, R13D
  call restar_c

  ;el resultado final ya está en EAX, así que no hay que hacer más nada

  ;epilogo
  pop R13 ;restauramos los registros no volátiles
  pop R12
  pop RBP ;pila desalineada, RBP restaurado, RSP apuntando a la dirección de retorno
  ret


alternate_sum_4_using_c_alternative:
  ;prologo
  push RBP ;pila alineada
  mov RBP, RSP ;strack frame armado
  sub RSP, 16 ; muevo el tope de la pila 8 bytes para guardar x4, y 8 bytes para que quede alineada

  mov [RBP-8], RCX ; guardo x4 en la pila

  push RDX  ;preservo x3 en la pila, desalineandola
  sub RSP, 8 ;alineo
  call restar_c 
  add RSP, 8 ;restauro tope
  pop RDX ;recupero x3
  
  mov EDI, EAX
  mov ESI, EDX
  call sumar_c

  mov EDI, EAX
  mov ESI, [RBP - 8] ;leo x4 de la pila
  call restar_c

  ;el resultado final ya está en EAX, así que no hay que hacer más nada

  ;epilogo
  add RSP, 16 ;restauro tope de pila
  pop RBP ;pila desalineada, RBP restaurado, RSP apuntando a la dirección de retorno
  ret


; uint32_t alternate_sum_8(uint32_t x1, uint32_t x2, uint32_t x3, uint32_t x4, uint32_t x5, uint32_t x6, uint32_t x7, uint32_t x8);
; registros y pila: x1[EDI], x2[ESI], x3[EDX], x4[ECX], x5[R8D], x6[R9D], x7[RBP + 0x10], x8[RBP + 0x18]
alternate_sum_8:
	;prologo
  push RBP
  mov RBP, RSP ; stack frame creado

  push R10
  push R11 ; al ser 2 queda alineada la pila

  call alternate_sum_4
  mov R10D, EAX

  mov EDI, R8D
  mov ESI, R9D
  mov EDX, [RBP + 16]
  mov ECX, [RBP + 24]

  call alternate_sum_4
  mov R11D, EAX

  add R10D, R11D
  mov EAX, R10D

	;epilogo
  pop R11
  pop R10
  pop RBP
	ret


; SUGERENCIA: investigar uso de instrucciones para convertir enteros a floats y viceversa
;void product_2_f(uint32_t * destination, uint32_t x1, float f1);
;registros: destination[RDI], x1[ESI], f1[XMM0]
product_2_f:
  ;prologo
  push RBP
  mov RBP, RSP

  CVTSI2SS XMM1, ESI

  mulss XMM1, XMM0

  CVTSS2SI EAX, XMM1

  mov [RDI], EAX

  pop RBP
	ret

;void product_2_SS(double * destination, float f1, float f2);
;registros: destination[RDI], f1[XMM0], f2[XMM1]
product_2_SS:
  ;prologo
  push RBP
  mov RBP, RSP

  mulss XMM1, XMM0

  CVTSS2SD XMM1, XMM1

  movsd [RDI], XMM1

  pop RBP
	ret


;extern void product_9_f(double * destination
;, uint32_t x1, float f1, uint32_t x2, float f2, uint32_t x3, float f3, uint32_t x4, float f4
;, uint32_t x5, float f5, uint32_t x6, float f6, uint32_t x7, float f7, uint32_t x8, float f8
;, uint32_t x9, float f9);
;registros y pila: destination[rdi], x1[esi], f1[xmm0], x2[edx], f2[xmm1], x3[ecx], f3[xmm2], x4[r8d], f4[xmm3]
;	, x5[r9d], f5[xmm4], x6[rbp+16], f6[xmm5], x7[rbp+24], f7[xmm6], x8[rbp+32], f8[xmm7],
;	, x9[rbp+40], f9[rbp+48]
product_9_f:
	;prologo
	push rbp
	mov rbp, rsp

  ; convert and multiply pair #1
  cvtsi2sd xmm8, esi
  cvtss2sd xmm0, xmm0
  mulsd xmm0, xmm8

  ; convert and multiply pair #2
  cvtsi2sd xmm8, edx
  mulsd xmm0, xmm8

  
  cvtss2sd xmm8, xmm1
  mulsd xmm0, xmm8

  ; convert and multiply pair #3
  cvtsi2sd xmm8, ecx
  mulsd xmm0, xmm8


  cvtss2sd xmm8, xmm2
  mulsd xmm0, xmm8

  ; convert and multiply pair #4
  cvtsi2sd xmm8, r8d
  mulsd xmm0, xmm8

  
  cvtss2sd xmm8, xmm3
  mulsd xmm0, xmm8

  ; convert and multiply pair #5
  cvtsi2sd xmm8, r9d
  mulsd xmm0, xmm8

  
  cvtss2sd xmm8, xmm4
  mulsd xmm0, xmm8

  ; convert and multiply pair #6
  cvtsi2sd xmm8, [rbp+16]
  mulsd xmm0, xmm8

  
  cvtss2sd xmm8, xmm5
  mulsd xmm0, xmm8

  ; convert and multiply pair #7
  cvtsi2sd xmm8, [rbp+24]
  mulsd xmm0, xmm8

  
  cvtss2sd xmm8, xmm6
  mulsd xmm0, xmm8

  ; convert and multiply pair #8
  cvtsi2sd xmm8, [rbp+32]
  mulsd xmm0, xmm8

  
  cvtss2sd xmm8, xmm7
  mulsd xmm0, xmm8

  ; convert and multiply pair #9
  cvtsi2sd xmm8, [rbp+40]
  mulsd xmm0, xmm8

  
  cvtss2sd xmm8, [rbp+48]
  mulsd xmm0, xmm8
  
  ;move result to [rdi]
  movsd [RDI], xmm0

	; epilogo
	pop rbp
	ret

