import { IsString, IsOptional, IsNumber } from 'class-validator';

export class CreateSessionDto {
  @IsString()
  @IsOptional()
  sourceLang?: string;

  @IsString()
  @IsOptional()
  targetLang?: string;

  @IsString()
  filePath: string;

  @IsString()
  @IsOptional()
  fileName?: string;

  @IsNumber()
  @IsOptional()
  fileSize?: number;
}
