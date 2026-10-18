import {
    Config,
    Expect,
    PathVariables,
    RequestBody,
    RequestConfig,
    RequestMapping,
    RequestParam,
    Service
} from "axios-annotations";
import type {Servant, ServantAvatar} from "./types.ts";

export const config = new Config(
    {
        protocol: 'http',
        host: 'localhost',
        port: 8000,
    }
);

@RequestConfig(config)
export class ResourceService extends Service {

    @RequestMapping("/dir/{path}", "GET")
    @RequestParam("type", false)
    @PathVariables()
    listDir(path: string, type: string = '') {
        return Expect<{
            "path": string;
            "entries": {
                "name": string;
                "type": string;
                "size": number;
            }[]
        }>({
            path,
            type
        });
    }
}

@RequestConfig(config)
@RequestMapping("/api")
export class ServantsService extends Service {
    @RequestMapping("/servants/import/csv", "POST")
    @RequestBody()
    importCsv(file: File) {
        const formData = new FormData();
        formData.append("file", file);
        return Expect<{
            "imported": number;
            "total": number;
        }>({
            body: formData
        });
    }

    @RequestMapping("/servants/import/json", "POST")
    @RequestBody()
    importJson(file: File) {
        const formData = new FormData();
        formData.append("file", file);
        return Expect<{
            "imported": number;
            "total": number;
        }>({
            body: formData
        });
    }

    @RequestMapping("/servants", "GET")
    listServants() {
        return Expect<Servant[]>({});
    }

    @RequestMapping("/avatars", "GET")
    listAllServantAvatars() {
        return Expect<ServantAvatar[]>({});
    }

    @RequestMapping("/servants/{servant_id}/avatars", "GET")
    @PathVariables()
    getServantAvatars(servant_id: number) {
        return Expect<ServantAvatar[]>({servant_id});
    }

    @RequestMapping("/detect/avatar-range/{servant_id}", "GET")
    @PathVariables()
    @RequestParam("illustration", true)
    @RequestParam("avatar", false)
    detectAvatar(servant_id: number,
                 illustration: string,
                 avatar: string | null) {
        return Expect<{
            "scale": number;
            "template_size": [number, number];
            "score": number;
            "x": number;
            "y": number;
            "w": number;
            "h": number;
            "rect": [number, number, number, number];
        } | null>({
            servant_id,
            illustration,
            avatar
        });
    }
}

export const ApiManager = {
    resourceService: new ResourceService(),
    servantsService: new ServantsService(),
}