#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>

int main() {
	int server_socket = socket(AF_UNIX, SOCK_STREAM, 0);

    struct sockaddr_un server_address;
    server_address.sun_family = AF_UNIX;
    strcpy(server_address.sun_path, "unix_socket");

	int connection = connect(server_socket, (struct sockaddr*)&server_address, sizeof(server_address));

	const char expresion[100];
	int resultado;

	scanf("%99s", &expresion);
	//fgets(expresion, sizeof(expresion), stdin); 
	while(strcmp(expresion, "exit") != 0){
        send(server_socket, &expresion, sizeof(expresion), 0);
		recv(server_socket, &resultado, sizeof(resultado), 0);
		printf("resultado de la operacion: %d\n", resultado);
		scanf("%s", &expresion);
	}
    send(server_socket, &expresion, sizeof(expresion), 0);	

	//envia el exit y termina el proceso cliente
	printf("chauchis \n");
	exit(0);
}
