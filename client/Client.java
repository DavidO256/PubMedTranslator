package client.translator;

import java.io.IOException;
import java.net.Socket;
import java.util.Arrays;

public class Client {

    private static String[] getTranslation(String ip, int port, String[] pmid) {
        String[] translations = new String[pmid.length];
        try {
            Socket socket = new Socket(ip, port);
            for(int i = 0; i < pmid.length; i++) {
                socket.getOutputStream().write(pmid[i].getBytes());
                while (socket.getInputStream().available() == 0) {
                       // zzz
                }
                byte[] raw = new byte[socket.getInputStream().available()];
                socket.getInputStream().read(raw);
                translations[i] = new String(raw);
            }
            socket.close();
        } catch (IOException e) { }
        return translations;
    }

    public static void main(String[] args) {
        if(args.length > 2) {
            String ip = args[0];
            int port = Integer.parseInt(args[1]);
            String[] pmid = Arrays.copyOfRange(args, 2, args.length);
            System.out.println(String.join("\n", getTranslation(ip, port, pmid)));
        } else
            System.err.println("Missing parameters!\nUsage: java -jar client.jar IP Port PMID#1 PMID#2 ... PMID#N");
    }
}