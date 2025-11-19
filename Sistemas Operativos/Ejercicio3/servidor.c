#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>

int calcular(const char *expresion) {
    int num1, num2, resultado;
    char operador;
    printf("Recibi: %s\n", expresion);
    // Usamos sscanf para extraer los dos números y el operador de la expresión
    if (sscanf(expresion, "%d%c%d", &num1, &operador, &num2) != 3) {
        printf("Formato incorrecto\n");
        return 0;  // En caso de error, retornamos 0.
    }

    // Realizamos la operación según el operador
    switch (operador) {
        case '+':
            resultado = num1 + num2;
            break;
        case '-':
            resultado = num1 - num2;
            break;
        case '*':
            resultado = num1 * num2;
            break;
        case '/':
            if (num2 != 0) {
                resultado = num1 / num2;
            } else {
                printf("Error: División por cero\n");
                return 0;  // Si hay división por cero, retornamos 0.
            }
            break;
        default:
            printf("Operador no reconocido\n");
            return 0;  // Si el operador no es válido, retornamos 0.
    }

    return resultado;
}

int main() {
     
    //configurar servidor
    int server_socket = socket(AF_UNIX, SOCK_STREAM, 0);

    struct sockaddr_un server_address;
    server_address.sun_family = AF_UNIX;
    strcpy(server_address.sun_path, "unix_socket");
    unlink(server_address.sun_path); //borra cualquier archivo con ese nombre en esa direccion 

    bind(server_socket, (struct sockaddr*)&server_address, sizeof(server_address));

    listen(server_socket, 1);

    struct sockaddr_un client_address;
    int client_address_size = sizeof(client_address);
    const char expresion[100];
    while(1){
        int client_socket = accept(server_socket, &client_address, &client_address_size);
        pid_t pid = fork();

        if(pid == 0){
            int resultado;
            recv(client_socket, &expresion, sizeof(expresion), 0);
            while(strcmp(expresion, "exit") != 0){
                resultado = calcular(&expresion);
                send(client_socket, &resultado, sizeof(resultado), 0);
                recv(client_socket, &expresion, sizeof(expresion), 0);
                //ssi recibo un SIGINT, me muero? o hay que definir la syscall
            }
            printf("me dijeron exit, Thank you forever\n");
            exit(0);
        }
        //hay que borrar a los hijos muertos        
    }

    // COMPLETAR. Este es un ejemplo de funcionamiento básico.
    // La expresión debe ser recibida como un mensaje del cliente hacia el servidor.
    /*const char *expresion = "10+5";  
    int resultado = calcular(expresion);
    printf("El resultado de la operación es: %d\n", resultado);*/
    exit(0);
}

