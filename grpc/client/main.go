package main

import (
	"context"
	"fmt"
	"log"

	pb "asdf/example"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
)

var (
	modelID   	  = "" // TODO: add your Baseten Model ID here
	basetenApiKey = "" // TODO: add your Baseten API key here
)

func main() {
	do := func() {
		conn, err := grpc.NewClient(fmt.Sprintf("model-%s.grpc.api.baseten.co:80", modelID), grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			log.Printf("could not connect: %v", err)
			return
		}
		defer conn.Close()

		client := pb.NewExampleServiceClient(conn)

		md := metadata.New(map[string]string{
			"baseten-authorization": fmt.Sprintf("Api-Key %s", basetenApiKey),
			"baseten-model-id":      fmt.Sprintf("model-%s", modelID),
		})

		// Create a new context with the metadata
		ctx := metadata.NewOutgoingContext(context.Background(), md)

		stream, err := client.Example(ctx, &pb.ExampleRequest{Message: "😊"})
		if err != nil {
			log.Printf("error calling example: %v", err)
			return
		}

		for {
			msg, err := stream.Recv()
			if err != nil {
				log.Printf("err: %v", err)

				return
			}
			log.Print(msg.Reply)
		}
	}

	do()
}
